// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subject-bound qualification evidence for humanoid motor authority.
//!
//! Internal qualification is deliberately distinct from legal/product safety
//! certification. This module answers a narrower runtime question: does the
//! current evidence corpus authorize goal-directed execution for this exact
//! morphology, task, actuation semantics, and backend profile?

use serde::{Deserialize, Serialize};

use crate::embodied_certification::EmbodiedControlCertificate;
use crate::execution::HumanoidAuthorityEnvelope;
use crate::morphology::HumanoidMorphology;
use crate::release_pipeline::{
    evaluate_release_pipeline, HumanoidReleasePipelinePolicy, HumanoidReleasePipelineReport,
    ReleaseStageKind, HUMANOID_RELEASE_PIPELINE_SCHEMA_VERSION,
};
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_QUALIFICATION_SUBJECT_SCHEMA_VERSION: u32 = 1;

/// Minimum release stages required before evidence can grant motor authority.
/// A custom release policy may be stricter, but not weaker, than this floor.
pub const MOTOR_AUTHORITY_REQUIRED_STAGES: [ReleaseStageKind; 8] = [
    ReleaseStageKind::SourceIntegrity,
    ReleaseStageKind::WorkspaceTests,
    ReleaseStageKind::MujocoOracleReproducibility,
    ReleaseStageKind::DynamicsOracleQualification,
    ReleaseStageKind::SparseSolverQualification,
    ReleaseStageKind::RealtimeQualification,
    ReleaseStageKind::HilFaultCampaign,
    ReleaseStageKind::SignedSafetyLedger,
];

/// Canonical subject of one humanoid qualification claim.
///
/// Qualification is intentionally task-specific. Evidence for `Stand` does not
/// imply evidence for `Run`, `Reach`, or `Grasp`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidQualificationSubject {
    pub schema_version: u32,
    pub morphology: HumanoidMorphology,
    pub task: HumanoidTask,
    pub actuation_mode: ActuationMode,
    /// Stable backend/hardware profile identity, not a display name.
    pub backend_profile_id: String,
}

impl HumanoidQualificationSubject {
    pub fn new(
        morphology: HumanoidMorphology,
        task: HumanoidTask,
        actuation_mode: ActuationMode,
        backend_profile_id: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: HUMANOID_QUALIFICATION_SUBJECT_SCHEMA_VERSION,
            morphology,
            task,
            actuation_mode,
            backend_profile_id: backend_profile_id.into(),
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == HUMANOID_QUALIFICATION_SUBJECT_SCHEMA_VERSION
            && !self.backend_profile_id.trim().is_empty()
            && self.backend_profile_id.len() <= 256
            && self
                .backend_profile_id
                .bytes()
                .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
    }

    /// Stable FNV-1a fingerprint over an explicitly versioned canonical encoding.
    /// This is an identity checksum, not a cryptographic signature.
    pub fn fingerprint(&self) -> u64 {
        if !self.validate() {
            return 0;
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        feed_u64(&mut hash, self.schema_version as u64);
        feed_u64(&mut hash, morphology_id(self.morphology));
        feed_u64(&mut hash, task_id(self.task));
        feed_u64(&mut hash, actuation_mode_id(self.actuation_mode));
        feed_bytes(&mut hash, self.backend_profile_id.as_bytes());
        if hash == 0 { 1 } else { hash }
    }
}

/// Runtime decision derived only after release evidence and embodied-control
/// evidence have been revalidated against the same qualification subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidQualificationEnvelope {
    pub subject_fingerprint: u64,
    pub release_id: String,
    pub policy_id: String,
    pub scenario_fingerprint: u64,
    pub evaluated_unix_millis: u64,
    pub qualified: bool,
    pub failures: Vec<String>,
}

impl HumanoidQualificationEnvelope {
    /// Qualification is binary at this layer. We do not reinterpret pass/fail
    /// evidence as a fabricated probability of competence.
    pub fn qualification_authority(&self) -> f32 {
        if self.qualified { 1.0 } else { 0.0 }
    }

    /// Tighten an already established authority envelope. Qualification cannot
    /// increase authority granted by another source.
    pub fn restrict_authority(
        &self,
        mut authority: HumanoidAuthorityEnvelope,
    ) -> HumanoidAuthorityEnvelope {
        let current = if authority.qualification.is_finite() {
            authority.qualification.clamp(0.0, 1.0)
        } else {
            0.0
        };
        authority.qualification = current.min(self.qualification_authority());
        authority
    }
}

/// Revalidate release evidence at `now_unix_millis`, bind it to the exact
/// subject, and combine it with the scenario/control certificate.
///
/// This function does not trust a historical `release.passed` snapshot. It
/// re-runs the release pipeline with the supplied current policy/time so evidence
/// that has become stale cannot retain qualification authority indefinitely.
pub fn evaluate_humanoid_qualification(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReleasePipelinePolicy,
    release: &HumanoidReleasePipelineReport,
    certificate: &EmbodiedControlCertificate,
    now_unix_millis: u64,
) -> HumanoidQualificationEnvelope {
    let subject_fingerprint = subject.fingerprint();
    let mut failures = Vec::new();

    if subject_fingerprint == 0 {
        failures.push("qualification subject is invalid".to_string());
    }
    if release.schema_version != HUMANOID_RELEASE_PIPELINE_SCHEMA_VERSION {
        failures.push("release pipeline schema is not supported".to_string());
    }
    if release.release_id.trim().is_empty() || release.policy_id != policy.policy_id {
        failures.push("release identity or policy binding does not match".to_string());
    }
    if release.subject_fingerprint == 0 || release.subject_fingerprint != subject_fingerprint {
        failures.push("release evidence is not bound to the requested subject".to_string());
    }
    if certificate.subject_fingerprint == 0
        || certificate.subject_fingerprint != subject_fingerprint
    {
        failures.push("control certificate is not bound to the requested subject".to_string());
    }
    if !certificate.accepted || certificate.total_cases == 0 {
        failures.push("embodied-control certificate was not accepted".to_string());
    }
    if certificate.scenario_fingerprint == 0 {
        failures.push("embodied-control scenario fingerprint is invalid".to_string());
    }
    if now_unix_millis == 0 {
        failures.push("qualification evaluation time is invalid".to_string());
    }

    for required in MOTOR_AUTHORITY_REQUIRED_STAGES {
        if !policy.required_stages.contains(&required) {
            failures.push(format!(
                "release policy is too weak for motor authority: missing {required:?}"
            ));
        }
    }

    // Re-evaluate the original stage corpus at the current time. This applies the
    // policy's freshness bound now rather than trusting the historical report.
    let current_release = evaluate_release_pipeline(
        release.release_id.clone(),
        policy,
        release.stages.clone(),
        now_unix_millis,
    );
    if !current_release.passed {
        failures.extend(
            current_release
                .failures
                .iter()
                .map(|failure| format!("current release evidence: {failure}")),
        );
    }
    if current_release.subject_fingerprint != subject_fingerprint {
        failures.push("current release revalidation changed subject identity".to_string());
    }

    HumanoidQualificationEnvelope {
        subject_fingerprint,
        release_id: release.release_id.clone(),
        policy_id: policy.policy_id.clone(),
        scenario_fingerprint: certificate.scenario_fingerprint,
        evaluated_unix_millis: now_unix_millis,
        qualified: failures.is_empty(),
        failures,
    }
}

fn feed_u64(hash: &mut u64, value: u64) {
    feed_bytes(hash, &value.to_le_bytes());
}

fn feed_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Delimiter prevents concatenation ambiguity.
    *hash ^= 0xff;
    *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
}

const fn morphology_id(value: HumanoidMorphology) -> u64 {
    match value {
        HumanoidMorphology::Dmc21 => 1,
        HumanoidMorphology::Dexterous53 => 2,
        HumanoidMorphology::FullSpine => 3,
        HumanoidMorphology::WithNeckWrist => 4,
    }
}

const fn task_id(value: HumanoidTask) -> u64 {
    match value {
        HumanoidTask::Stand => 1,
        HumanoidTask::Walk => 2,
        HumanoidTask::Run => 3,
        HumanoidTask::Reach => 4,
        HumanoidTask::Grasp => 5,
    }
}

const fn actuation_mode_id(value: ActuationMode) -> u64 {
    match value {
        ActuationMode::NormalizedTorque => 1,
        ActuationMode::TorqueNewtonMetres => 2,
        ActuationMode::NormalizedPosition => 3,
        ActuationMode::PositionTargetRadians => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_fingerprint_changes_with_task() {
        let stand = HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Stand,
            ActuationMode::NormalizedTorque,
            "simple-sim-v1",
        );
        let run = HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Run,
            ActuationMode::NormalizedTorque,
            "simple-sim-v1",
        );
        assert_ne!(stand.fingerprint(), 0);
        assert_ne!(stand.fingerprint(), run.fingerprint());
    }

    #[test]
    fn invalid_subject_fails_closed() {
        let subject = HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Stand,
            ActuationMode::NormalizedTorque,
            "",
        );
        assert_eq!(subject.fingerprint(), 0);
    }

    #[test]
    fn failed_qualification_cannot_raise_existing_authority() {
        let envelope = HumanoidQualificationEnvelope {
            subject_fingerprint: 1,
            release_id: "r".to_string(),
            policy_id: "p".to_string(),
            scenario_fingerprint: 1,
            evaluated_unix_millis: 1,
            qualified: false,
            failures: vec!["failed".to_string()],
        };
        let authority = envelope.restrict_authority(HumanoidAuthorityEnvelope {
            qualification: 0.2,
            ..HumanoidAuthorityEnvelope::fully_admitted()
        });
        assert_eq!(authority.qualification, 0.0);
    }
}
