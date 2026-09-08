// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifier-owned positive witness for current humanoid qualification.
//!
//! `HumanoidQualificationEnvelope` is useful serializable audit data, but its
//! public fields and Serde surface mean deserialized data must not itself stand
//! in for a freshly executed verifier. This module adds a process-local positive
//! witness whose only constructor re-runs the current subject/release/control
//! qualification checks.
//!
//! The compatibility envelope remains unchanged. Future physical-runtime code
//! can migrate to requiring `VerifiedHumanoidQualification` without forcing an
//! immediate wire-format or persistence migration.

use crate::embodied_certification::EmbodiedControlCertificate;
use crate::execution::HumanoidAuthorityEnvelope;
use crate::qualification::{
    HumanoidQualificationEnvelope, HumanoidQualificationSubject, evaluate_humanoid_qualification,
};
use crate::release_pipeline::{HumanoidReleasePipelinePolicy, HumanoidReleasePipelineReport};

/// Positive, current qualification evidence owned by the verifier boundary.
///
/// Intentionally not `Clone`, `Copy`, `Serialize`, or `Deserialize`. Persisted
/// envelopes are audit data and must pass the verifier again to recreate this
/// witness after restart or evidence-currentness changes.
#[derive(Debug)]
pub struct VerifiedHumanoidQualification {
    envelope: HumanoidQualificationEnvelope,
}

impl VerifiedHumanoidQualification {
    pub fn audit_record(&self) -> &HumanoidQualificationEnvelope {
        &self.envelope
    }

    pub const fn subject_fingerprint(&self) -> u64 {
        self.envelope.subject_fingerprint
    }

    pub fn release_id(&self) -> &str {
        &self.envelope.release_id
    }

    pub fn policy_id(&self) -> &str {
        &self.envelope.policy_id
    }

    pub const fn scenario_fingerprint(&self) -> u64 {
        self.envelope.scenario_fingerprint
    }

    pub const fn evaluated_unix_millis(&self) -> u64 {
        self.envelope.evaluated_unix_millis
    }

    /// A verified qualification can only tighten an already-composed authority
    /// envelope. It does not manufacture authority that another independent
    /// source has already restricted.
    pub fn restrict_authority(
        &self,
        authority: HumanoidAuthorityEnvelope,
    ) -> HumanoidAuthorityEnvelope {
        self.envelope.restrict_authority(authority)
    }
}

/// Re-run the complete current humanoid qualification verifier and return a
/// process-local positive witness only when the resulting envelope is qualified.
///
/// On failure the serializable envelope is returned so callers retain exact
/// diagnostic/audit evidence without receiving a positive runtime capability.
pub fn verify_current_humanoid_qualification(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReleasePipelinePolicy,
    release: &HumanoidReleasePipelineReport,
    certificate: &EmbodiedControlCertificate,
    now_unix_millis: u64,
) -> Result<VerifiedHumanoidQualification, HumanoidQualificationEnvelope> {
    let envelope = evaluate_humanoid_qualification(
        subject,
        policy,
        release,
        certificate,
        now_unix_millis,
    );
    if envelope.qualified {
        Ok(VerifiedHumanoidQualification { envelope })
    } else {
        Err(envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodied_certification::EmbodiedControlCertificate;
    use crate::morphology::HumanoidMorphology;
    use crate::qualification::{
        MOTOR_AUTHORITY_REQUIRED_STAGES, SUPPORTED_EMBODIED_CONTROL_CERTIFICATE_SCHEMA_VERSION,
    };
    use crate::release_pipeline::{ReleaseStageEvidence, evaluate_release_pipeline};
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            task,
            ActuationMode::NormalizedTorque,
            "verified-qualification-test-v1",
        )
    }

    fn release_for(
        subject: &HumanoidQualificationSubject,
        generated_at: u64,
    ) -> HumanoidReleasePipelineReport {
        let fingerprint = subject.fingerprint();
        let stages = MOTOR_AUTHORITY_REQUIRED_STAGES
            .iter()
            .copied()
            .enumerate()
            .map(|(index, stage)| ReleaseStageEvidence {
                stage,
                artifact_id: format!("artifact-{index}"),
                artifact_sha256: format!("{:064x}", index + 1),
                producer_id: format!("producer-{index}"),
                passed: true,
                generated_unix_millis: generated_at,
                subject_fingerprint: fingerprint,
            })
            .collect();
        evaluate_release_pipeline(
            "verified-release-1",
            &HumanoidReleasePipelinePolicy::default(),
            stages,
            generated_at,
        )
    }

    fn certificate_for(subject: &HumanoidQualificationSubject) -> EmbodiedControlCertificate {
        EmbodiedControlCertificate {
            schema_version: SUPPORTED_EMBODIED_CONTROL_CERTIFICATE_SCHEMA_VERSION,
            subject_fingerprint: subject.fingerprint(),
            scenario_fingerprint: 42,
            total_cases: 10,
            fallback_rate: 0.0,
            budget_miss_rate: 0.0,
            fall_rate: 0.0,
            recovery_rate: 1.0,
            maximum_terrain_height_std_m: 0.0,
            maximum_terrain_evidence_age_s: 0.0,
            solver_derived_cases: 10,
            floating_base_cases: 10,
            upper_body_contact_cases: 1,
            accepted: true,
            failures: Vec::new(),
        }
    }

    #[test]
    fn current_valid_evidence_produces_process_local_positive_witness() {
        let subject = subject(HumanoidTask::Stand);
        let policy = HumanoidReleasePipelinePolicy::default();
        let release = release_for(&subject, 1_000);
        let certificate = certificate_for(&subject);
        let verified = verify_current_humanoid_qualification(
            &subject,
            &policy,
            &release,
            &certificate,
            1_000,
        )
        .unwrap();
        assert!(verified.audit_record().qualified);
        assert_eq!(verified.subject_fingerprint(), subject.fingerprint());
        assert_eq!(verified.release_id(), "verified-release-1");
        assert_eq!(verified.scenario_fingerprint(), 42);
    }

    #[test]
    fn stale_evidence_returns_audit_failure_without_positive_witness() {
        let subject = subject(HumanoidTask::Stand);
        let policy = HumanoidReleasePipelinePolicy::default();
        let release = release_for(&subject, 1_000);
        let certificate = certificate_for(&subject);
        let stale_now = 1_000 + policy.maximum_evidence_age_millis + 1;
        let failure = verify_current_humanoid_qualification(
            &subject,
            &policy,
            &release,
            &certificate,
            stale_now,
        )
        .unwrap_err();
        assert!(!failure.qualified);
        assert!(failure
            .failures
            .iter()
            .any(|reason| reason.contains("stale evidence")));
    }

    #[test]
    fn wrong_task_cannot_produce_positive_witness() {
        let stand = subject(HumanoidTask::Stand);
        let run = subject(HumanoidTask::Run);
        let policy = HumanoidReleasePipelinePolicy::default();
        let release = release_for(&stand, 1_000);
        let certificate = certificate_for(&stand);
        let failure = verify_current_humanoid_qualification(
            &run,
            &policy,
            &release,
            &certificate,
            1_000,
        )
        .unwrap_err();
        assert!(!failure.qualified);
    }

    #[test]
    fn verified_qualification_cannot_raise_a_stricter_existing_limit() {
        let subject = subject(HumanoidTask::Stand);
        let policy = HumanoidReleasePipelinePolicy::default();
        let release = release_for(&subject, 1_000);
        let certificate = certificate_for(&subject);
        let verified = verify_current_humanoid_qualification(
            &subject,
            &policy,
            &release,
            &certificate,
            1_000,
        )
        .unwrap();
        let authority = HumanoidAuthorityEnvelope {
            qualification: 0.2,
            ..HumanoidAuthorityEnvelope::fully_admitted()
        };
        assert_eq!(verified.restrict_authority(authority).qualification, 0.2);
    }
}
