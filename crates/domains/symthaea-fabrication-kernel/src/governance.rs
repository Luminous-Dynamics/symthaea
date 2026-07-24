// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! High-level governance ceremony for trusted fabrication transitions.
//!
//! Lower-level APIs remain available, but this orchestrator makes lifecycle
//! verification and audit recording inseparable for the common release path.

use crate::attestation::{
    AttestationPolicy, AttestationTrustContext, AttestationVerificationReport,
    AttestedFabricationManifest, ManifestSignatureVerifier, VerifiedAttestation,
    verify_attestation_authority_with_trust,
};
use crate::audit::{AuditAction, AuditAppendError, AuditJournal};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::execution_guard::{ContainmentAction, GuardDecision};
use crate::fault_injection::FaultInjectionReport;
use crate::machine::{NegotiatedMachine, TimedMachineSession, ValidatedGCode};
use crate::operational_replay::{OperationalReplayError, digest_fault_injection_matrix};
use crate::operator_command::{
    OperatorCommandExpectation, OperatorCommandPolicy, OperatorCommandVerifier,
    OperatorCommandViolation, SignedOperatorCommand, verify_operator_command,
};
use crate::operator_command_tracker::{
    AppliedOperatorCommand, OperatorCommandTracker, OperatorCommandTrackingError,
};
use crate::release::{
    ReleaseAuthority, ReleaseAuthorization, ReleaseEvaluationReport, ReleasePolicy,
    authorize_release,
};
use crate::rotation::{
    AuthorizedTrustRotation, KeyRotationPolicy, SignedTrustRotationProposal, TrustRotationVerifier,
    TrustRotationViolation, authorize_trust_rotation,
};
use crate::session::{MachineSessionTracker, SessionTrackingError};
use crate::submission::{
    AuthorizedPrintJob, GovernedAuthorizedPrintJob, GovernedPrintAuthorizationError,
    GovernedSubmittedJobReceipt, PrintAuthorizationError, SubmittedJobReceipt,
    authorize_governed_print_job, authorize_print_job,
};
use crate::trust::{
    TrustSnapshot, TrustSnapshotError, TrustSnapshotTracker, TrustSnapshotTrackingError,
    digest_trust_snapshot,
};

#[derive(Debug, Clone, PartialEq)]
pub enum GovernanceError {
    InvalidActor,
    ActorTooLong,
    TrustSnapshot(TrustSnapshotError),
    TrustSnapshotStale,
    Attestation(AttestationVerificationReport),
    PrintAuthorization(PrintAuthorizationError),
    Release(ReleaseEvaluationReport),
    Session(SessionTrackingError),
    TrustRotation(Vec<TrustRotationViolation>),
    TrustRotationTracking(TrustSnapshotTrackingError),
    TrustRotationCurrentMismatch,
    TrustRotationNotYetActive,
    TrustRotationExpired,
    GovernedPrintAuthorization(GovernedPrintAuthorizationError),
    OperationalReplay(OperationalReplayError),
    OperatorCommand(Vec<OperatorCommandViolation>),
    OperatorCommandTracking(OperatorCommandTrackingError),
    Audit(AuditAppendError),
}

pub struct FabricationGovernance {
    actor: String,
    evaluation_time_unix_s: u64,
    trust_snapshot: TrustSnapshot,
    trust_snapshot_digest: Sha256Digest,
    audit_journal: AuditJournal,
    session_tracker: MachineSessionTracker,
    trust_tracker: TrustSnapshotTracker,
    operator_command_tracker: OperatorCommandTracker,
}

impl FabricationGovernance {
    pub fn new(
        actor: impl Into<String>,
        evaluation_time_unix_s: u64,
        trust_snapshot: TrustSnapshot,
    ) -> Result<Self, GovernanceError> {
        let actor = actor.into();
        if actor.trim().is_empty() {
            return Err(GovernanceError::InvalidActor);
        }
        if actor.len() > crate::audit::MAX_AUDIT_ACTOR_BYTES {
            return Err(GovernanceError::ActorTooLong);
        }
        trust_snapshot
            .validate()
            .map_err(GovernanceError::TrustSnapshot)?;
        if !trust_snapshot.is_fresh_at(evaluation_time_unix_s) {
            return Err(GovernanceError::TrustSnapshotStale);
        }
        let trust_snapshot_digest =
            digest_trust_snapshot(&trust_snapshot).map_err(GovernanceError::TrustSnapshot)?;
        let mut trust_tracker = TrustSnapshotTracker::default();
        trust_tracker
            .accept(&trust_snapshot)
            .map_err(GovernanceError::TrustRotationTracking)?;
        Ok(Self {
            actor,
            evaluation_time_unix_s,
            trust_snapshot,
            trust_snapshot_digest,
            audit_journal: AuditJournal::default(),
            session_tracker: MachineSessionTracker::default(),
            trust_tracker,
            operator_command_tracker: OperatorCommandTracker::default(),
        })
    }

    pub fn authorize_trust_rotation(
        &mut self,
        timestamp_unix_s: u64,
        signed: SignedTrustRotationProposal,
        policy: &KeyRotationPolicy,
        verifier: &dyn TrustRotationVerifier,
    ) -> Result<AuthorizedTrustRotation, GovernanceError> {
        let authorized = authorize_trust_rotation(
            signed,
            &self.trust_snapshot,
            policy,
            timestamp_unix_s,
            verifier,
        )
        .map_err(GovernanceError::TrustRotation)?;
        let details = digest_rotation_authorization_context(&authorized);
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::TrustRotationAuthorized,
                authorized.proposed_snapshot_digest(),
                Some(details),
            )
            .map_err(GovernanceError::Audit)?;
        Ok(authorized)
    }

    /// Atomically activate an already authorized rotation in memory.
    pub fn activate_trust_rotation(
        &mut self,
        timestamp_unix_s: u64,
        authorized: AuthorizedTrustRotation,
    ) -> Result<(), GovernanceError> {
        if authorized.current_snapshot_digest() != self.trust_snapshot_digest {
            return Err(GovernanceError::TrustRotationCurrentMismatch);
        }
        if timestamp_unix_s < authorized.activates_at_unix_s() {
            return Err(GovernanceError::TrustRotationNotYetActive);
        }
        let proposed = authorized.proposal().proposed_snapshot.clone();
        if !proposed.is_fresh_at(timestamp_unix_s) {
            return Err(GovernanceError::TrustRotationExpired);
        }

        let mut next_tracker = self.trust_tracker.clone();
        next_tracker
            .accept(&proposed)
            .map_err(GovernanceError::TrustRotationTracking)?;
        let mut next_audit = self.audit_journal.clone();
        next_audit
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::TrustRotationActivated,
                authorized.proposed_snapshot_digest(),
                Some(authorized.proposal_digest()),
            )
            .map_err(GovernanceError::Audit)?;

        self.trust_snapshot_digest = authorized.proposed_snapshot_digest();
        self.trust_snapshot = proposed;
        self.evaluation_time_unix_s = timestamp_unix_s;
        self.trust_tracker = next_tracker;
        self.audit_journal = next_audit;
        Ok(())
    }

    pub fn verify_attestation(
        &mut self,
        timestamp_unix_s: u64,
        attested: AttestedFabricationManifest,
        policy: &AttestationPolicy,
        verifier: &dyn ManifestSignatureVerifier,
    ) -> Result<VerifiedAttestation, GovernanceError> {
        let verified = verify_attestation_authority_with_trust(
            attested,
            policy,
            verifier,
            AttestationTrustContext {
                evaluation_time_unix_s: self.evaluation_time_unix_s,
                snapshot: &self.trust_snapshot,
            },
        )
        .map_err(GovernanceError::Attestation)?;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::AttestationVerified,
                verified.manifest_digest(),
                Some(self.trust_snapshot_digest),
            )
            .map_err(GovernanceError::Audit)?;
        Ok(verified)
    }

    pub fn authorize_release(
        &mut self,
        timestamp_unix_s: u64,
        attestation: &VerifiedAttestation,
        policy: &ReleasePolicy,
    ) -> Result<ReleaseAuthorization, GovernanceError> {
        let authorization =
            authorize_release(attestation, policy).map_err(GovernanceError::Release)?;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::ReleaseAuthorized,
                authorization.manifest_digest(),
                Some(authorization.policy_digest()),
            )
            .map_err(GovernanceError::Audit)?;
        Ok(authorization)
    }

    pub fn accept_machine_session(
        &mut self,
        timestamp_unix_s: u64,
        session: &TimedMachineSession,
    ) -> Result<Sha256Digest, GovernanceError> {
        let digest = self
            .session_tracker
            .accept(session)
            .map_err(GovernanceError::Session)?;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::MachineSessionAccepted,
                digest,
                Some(
                    self.session_tracker
                        .digest()
                        .map_err(GovernanceError::Session)?,
                ),
            )
            .map_err(GovernanceError::Audit)?;
        Ok(digest)
    }

    pub fn authorize_governed_job(
        &mut self,
        timestamp_unix_s: u64,
        validated: ValidatedGCode,
        attestation: VerifiedAttestation,
        release: &dyn ReleaseAuthority,
        machine: NegotiatedMachine,
    ) -> Result<GovernedAuthorizedPrintJob, GovernanceError> {
        let lease = self
            .session_tracker
            .consume(&machine, timestamp_unix_s)
            .map_err(GovernanceError::Session)?;
        let session_digest = lease.session_digest();
        let manifest_digest = attestation.manifest_digest();
        let authorized = authorize_governed_print_job(
            validated,
            attestation,
            release,
            machine,
            lease,
            timestamp_unix_s,
        )
        .map_err(GovernanceError::GovernedPrintAuthorization)?;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::MachineSessionConsumed,
                session_digest,
                Some(release.policy_digest()),
            )
            .map_err(GovernanceError::Audit)?;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::JobAuthorized,
                manifest_digest,
                Some(digest_governed_authorization_context(&authorized)),
            )
            .map_err(GovernanceError::Audit)?;
        Ok(authorized)
    }

    pub fn authorize_job(
        &mut self,
        timestamp_unix_s: u64,
        validated: ValidatedGCode,
        attestation: VerifiedAttestation,
        machine: NegotiatedMachine,
    ) -> Result<AuthorizedPrintJob, GovernanceError> {
        let manifest_digest = attestation.manifest_digest();
        let context_digest = digest_authorization_context(
            machine.machine_id(),
            machine.session_nonce(),
            validated.profile_name(),
        );
        let authorized = authorize_print_job(validated, attestation, machine)
            .map_err(GovernanceError::PrintAuthorization)?;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::JobAuthorized,
                manifest_digest,
                Some(context_digest),
            )
            .map_err(GovernanceError::Audit)?;
        Ok(authorized)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_and_apply_operator_command(
        &mut self,
        timestamp_unix_ms: u64,
        signed: SignedOperatorCommand,
        policy: &OperatorCommandPolicy,
        manifest_digest: Sha256Digest,
        machine_id: &str,
        session_digest: Sha256Digest,
        printer_job_id: &str,
        verifier: &dyn OperatorCommandVerifier,
    ) -> Result<AppliedOperatorCommand, GovernanceError> {
        let verified = verify_operator_command(
            signed,
            policy,
            OperatorCommandExpectation {
                manifest_digest,
                machine_id,
                session_digest,
                printer_job_id,
                now_unix_ms: timestamp_unix_ms,
                trust_snapshot: &self.trust_snapshot,
            },
            verifier,
        )
        .map_err(GovernanceError::OperatorCommand)?;
        let command_digest = verified.command_digest();
        let applied = self
            .operator_command_tracker
            .apply(&verified)
            .map_err(GovernanceError::OperatorCommandTracking)?;
        let timestamp_unix_s = timestamp_unix_ms / 1_000;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::OperatorCommandVerified,
                manifest_digest,
                Some(command_digest),
            )
            .map_err(GovernanceError::Audit)?;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::OperatorCommandApplied,
                command_digest,
                Some(
                    self.operator_command_tracker
                        .digest()
                        .map_err(GovernanceError::OperatorCommandTracking)?,
                ),
            )
            .map_err(GovernanceError::Audit)?;
        Ok(applied)
    }

    pub fn record_submission(
        &mut self,
        timestamp_unix_s: u64,
        receipt: &SubmittedJobReceipt,
    ) -> Result<Sha256Digest, GovernanceError> {
        let details = digest_submission_context(receipt);
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::JobSubmitted,
                receipt.manifest_digest,
                Some(details),
            )
            .map_err(GovernanceError::Audit)
    }

    pub fn record_fault_injection_matrix(
        &mut self,
        timestamp_unix_s: u64,
        manifest_digest: Sha256Digest,
        reports: &[FaultInjectionReport],
    ) -> Result<Sha256Digest, GovernanceError> {
        let matrix_digest =
            digest_fault_injection_matrix(reports).map_err(GovernanceError::OperationalReplay)?;
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::FaultInjectionVerified,
                manifest_digest,
                Some(matrix_digest),
            )
            .map_err(GovernanceError::Audit)
    }

    pub fn record_governed_submission(
        &mut self,
        timestamp_unix_s: u64,
        receipt: &GovernedSubmittedJobReceipt,
    ) -> Result<Sha256Digest, GovernanceError> {
        let details = digest_governed_submission_context(receipt);
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                AuditAction::JobSubmitted,
                receipt.submission.manifest_digest,
                Some(details),
            )
            .map_err(GovernanceError::Audit)
    }

    pub fn record_containment(
        &mut self,
        timestamp_unix_s: u64,
        manifest_digest: Sha256Digest,
        decision: &GuardDecision,
    ) -> Result<Option<Sha256Digest>, GovernanceError> {
        let action = match decision.latched_action {
            ContainmentAction::Continue => return Ok(None),
            ContainmentAction::Pause => AuditAction::ExecutionPaused,
            ContainmentAction::Cancel => AuditAction::ExecutionCancelled,
            ContainmentAction::EmergencyStop => AuditAction::EmergencyStopped,
        };
        let details = digest_guard_decision(decision);
        self.audit_journal
            .append(
                timestamp_unix_s,
                self.actor.clone(),
                action,
                manifest_digest,
                Some(details),
            )
            .map(Some)
            .map_err(GovernanceError::Audit)
    }

    pub fn trust_snapshot(&self) -> &TrustSnapshot {
        &self.trust_snapshot
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn audit_journal(&self) -> &AuditJournal {
        &self.audit_journal
    }

    pub fn session_tracker(&self) -> &MachineSessionTracker {
        &self.session_tracker
    }

    pub fn trust_tracker(&self) -> &TrustSnapshotTracker {
        &self.trust_tracker
    }

    pub fn operator_command_tracker(&self) -> &OperatorCommandTracker {
        &self.operator_command_tracker
    }

    pub fn into_operational_evidence(self) -> (TrustSnapshot, AuditJournal, MachineSessionTracker) {
        (
            self.trust_snapshot,
            self.audit_journal,
            self.session_tracker,
        )
    }

    pub fn into_operational_evidence_with_commands(
        self,
    ) -> (
        TrustSnapshot,
        AuditJournal,
        MachineSessionTracker,
        OperatorCommandTracker,
    ) {
        (
            self.trust_snapshot,
            self.audit_journal,
            self.session_tracker,
            self.operator_command_tracker,
        )
    }

    pub fn into_evidence(self) -> (TrustSnapshot, AuditJournal) {
        (self.trust_snapshot, self.audit_journal)
    }
}

fn digest_authorization_context(
    machine_id: &str,
    session_nonce: &str,
    profile_name: &str,
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.authorization-context.v1\0");
    append_string(&mut hasher, machine_id);
    append_string(&mut hasher, session_nonce);
    append_string(&mut hasher, profile_name);
    hasher.finalize()
}

fn digest_submission_context(receipt: &SubmittedJobReceipt) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.submission-context.v1\0");
    append_string(&mut hasher, &receipt.printer_job_id);
    append_string(&mut hasher, &receipt.machine_id);
    append_string(&mut hasher, &receipt.session_nonce);
    hasher.finalize()
}

fn digest_guard_decision(decision: &GuardDecision) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.guard-decision.v1\0");
    hasher.update(&[decision.action as u8, decision.latched_action as u8]);
    hasher.update(&(decision.new_violations.len() as u64).to_le_bytes());
    for violation in &decision.new_violations {
        append_string(&mut hasher, &format!("{violation:?}"));
    }
    hasher.finalize()
}

fn digest_governed_authorization_context(job: &GovernedAuthorizedPrintJob) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.governed-authorization-context.v1\0");
    hasher.update(&job.release_policy_digest().0);
    hasher.update(&job.trust_snapshot_digest().0);
    hasher.update(&job.session_digest().0);
    hasher.update(&job.session_sequence().to_le_bytes());
    if let Some(digest) = job.delegation_digest() {
        hasher.update(&[1]);
        hasher.update(&digest.0);
    } else {
        hasher.update(&[0]);
    }
    hasher.finalize()
}

fn digest_governed_submission_context(receipt: &GovernedSubmittedJobReceipt) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.governed-submission-context.v1\0");
    let base = digest_submission_context(&receipt.submission);
    hasher.update(&base.0);
    hasher.update(&receipt.release_policy_digest.0);
    hasher.update(&receipt.trust_snapshot_digest.0);
    hasher.update(&receipt.session_digest.0);
    hasher.update(&receipt.session_sequence.to_le_bytes());
    if let Some(digest) = receipt.delegation_digest {
        hasher.update(&[1]);
        hasher.update(&digest.0);
    } else {
        hasher.update(&[0]);
    }
    hasher.finalize()
}

fn digest_rotation_authorization_context(authorized: &AuthorizedTrustRotation) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.rotation-authorization-context.v1\0");
    hasher.update(&authorized.current_snapshot_digest().0);
    hasher.update(&authorized.proposed_snapshot_digest().0);
    hasher.update(&authorized.proposal_digest().0);
    hasher.update(&authorized.policy_digest().0);
    hasher.update(&authorized.activates_at_unix_s().to_le_bytes());
    hasher.update(&authorized.authorized_at_unix_s().to_le_bytes());
    hasher.finalize()
}

fn append_string(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::{ManifestSigner, SignatureAlgorithm, attest_fabrication_manifest};
    use crate::crypto_digest::sha256;
    use crate::provenance::{FabricationManifest, StableFingerprint};
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage};
    use std::collections::BTreeSet;

    struct TestProvider;

    impl ManifestSigner for TestProvider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Other("test".into())
        }
        fn key_id(&self) -> &str {
            "governance-key"
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
                && key_id == "governance-key"
                && signature == sha256(message).0.as_slice())
        }
    }

    fn manifest() -> FabricationManifest {
        let value = StableFingerprint([1, 2, 3, 4]);
        FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: value,
            process_policy: value,
            process_evidence: value,
            minimum_feature_policy: value,
            minimum_feature_evidence: value,
            slice_config: value,
            slice_layers: value,
            toolpath_config: value,
            machine_profile: value,
            gcode_program: value,
            pipeline: value,
            layer_count: 1,
            command_count: 1,
            total_extrusion_mm: 1.0,
        }
    }

    fn trust() -> TrustSnapshot {
        TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Other("test".into()),
                key_id: "governance-key".into(),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            }],
        )
        .unwrap()
    }

    #[test]
    fn verification_always_appends_governance_evidence() {
        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let mut governance = FabricationGovernance::new("operator", 500, trust()).unwrap();
        let verified = governance
            .verify_attestation(501, attested, &AttestationPolicy::default(), &provider)
            .unwrap();
        assert!(verified.is_lifecycle_governed());
        assert_eq!(governance.audit_journal().events.len(), 1);
        assert!(governance.audit_journal().verify().intact());
    }

    #[test]
    fn stale_snapshot_cannot_start_a_ceremony() {
        assert!(matches!(
            FabricationGovernance::new("operator", 1_000, trust()),
            Err(GovernanceError::TrustSnapshotStale)
        ));
    }
}
