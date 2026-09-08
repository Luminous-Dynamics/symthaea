// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Automated, fail-closed Humanoid release-certification pipeline state.
//!
//! Every stage is bound to one qualification-subject fingerprint. Mixing stage
//! evidence from different morphologies/tasks/backends is a release failure.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const HUMANOID_RELEASE_PIPELINE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseStageKind {
    SourceIntegrity,
    WorkspaceTests,
    MujocoOracleReproducibility,
    DynamicsOracleQualification,
    SparseSolverQualification,
    RealtimeQualification,
    HilFaultCampaign,
    SignedSafetyLedger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseStageEvidence {
    pub stage: ReleaseStageKind,
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub producer_id: String,
    pub passed: bool,
    pub generated_unix_millis: u64,
    /// Canonical `HumanoidQualificationSubject::fingerprint()` carried by the
    /// artifact. Legacy/unbound evidence deserializes as zero and fails closed.
    #[serde(default)]
    pub subject_fingerprint: u64,
}

impl ReleaseStageEvidence {
    pub fn validate(&self) -> bool {
        !self.artifact_id.trim().is_empty()
            && is_sha256(&self.artifact_sha256)
            && !self.producer_id.trim().is_empty()
            && self.generated_unix_millis > 0
            && self.subject_fingerprint != 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidReleasePipelinePolicy {
    pub policy_id: String,
    pub required_stages: Vec<ReleaseStageKind>,
    pub require_unique_producers: bool,
    pub maximum_evidence_age_millis: u64,
}

impl Default for HumanoidReleasePipelinePolicy {
    fn default() -> Self {
        Self {
            policy_id: "symthaea-humanoid-production-v1".to_string(),
            required_stages: vec![
                ReleaseStageKind::SourceIntegrity,
                ReleaseStageKind::WorkspaceTests,
                ReleaseStageKind::MujocoOracleReproducibility,
                ReleaseStageKind::DynamicsOracleQualification,
                ReleaseStageKind::SparseSolverQualification,
                ReleaseStageKind::RealtimeQualification,
                ReleaseStageKind::HilFaultCampaign,
                ReleaseStageKind::SignedSafetyLedger,
            ],
            require_unique_producers: false,
            maximum_evidence_age_millis: 7 * 24 * 60 * 60 * 1_000,
        }
    }
}

impl HumanoidReleasePipelinePolicy {
    pub fn validate(&self) -> bool {
        !self.policy_id.trim().is_empty()
            && !self.required_stages.is_empty()
            && self.required_stages.iter().collect::<BTreeSet<_>>().len()
                == self.required_stages.len()
            && self.maximum_evidence_age_millis > 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidReleasePipelineReport {
    pub schema_version: u32,
    pub release_id: String,
    pub policy_id: String,
    /// Common qualification subject shared by every admitted release stage.
    #[serde(default)]
    pub subject_fingerprint: u64,
    pub evaluated_unix_millis: u64,
    pub stages: Vec<ReleaseStageEvidence>,
    pub passed: bool,
    pub failures: Vec<String>,
}

pub fn evaluate_release_pipeline(
    release_id: impl Into<String>,
    policy: &HumanoidReleasePipelinePolicy,
    stages: Vec<ReleaseStageEvidence>,
    now_unix_millis: u64,
) -> HumanoidReleasePipelineReport {
    let release_id = release_id.into();
    let mut failures = Vec::new();
    if release_id.trim().is_empty() || !policy.validate() || now_unix_millis == 0 {
        failures.push("release identity, policy, or evaluation time is invalid".to_string());
    }
    let mut by_stage = BTreeMap::new();
    let mut artifact_ids = BTreeSet::new();
    let mut artifact_hashes = BTreeSet::new();
    let mut subject_fingerprint: Option<u64> = None;
    for evidence in &stages {
        if !evidence.validate() {
            failures.push(format!("invalid evidence for {:?}", evidence.stage));
            continue;
        }
        match subject_fingerprint {
            None => subject_fingerprint = Some(evidence.subject_fingerprint),
            Some(existing) if existing != evidence.subject_fingerprint => failures.push(format!(
                "stage {:?} is bound to subject {:016x}, expected {:016x}",
                evidence.stage, evidence.subject_fingerprint, existing
            )),
            Some(_) => {}
        }
        if by_stage.insert(evidence.stage, evidence).is_some() {
            failures.push(format!("duplicate evidence for {:?}", evidence.stage));
        }
        if !artifact_ids.insert(evidence.artifact_id.as_str()) {
            failures.push(format!(
                "duplicate artifact identity {}",
                evidence.artifact_id
            ));
        }
        if !artifact_hashes.insert(evidence.artifact_sha256.as_str()) {
            failures.push(format!("duplicate artifact hash for {:?}", evidence.stage));
        }
        if evidence.generated_unix_millis > now_unix_millis.saturating_add(1_000) {
            failures.push(format!("future-dated evidence for {:?}", evidence.stage));
        } else if now_unix_millis.saturating_sub(evidence.generated_unix_millis)
            > policy.maximum_evidence_age_millis
        {
            failures.push(format!("stale evidence for {:?}", evidence.stage));
        }
        if !evidence.passed {
            failures.push(format!("stage {:?} did not pass", evidence.stage));
        }
    }
    for required in &policy.required_stages {
        if !by_stage.contains_key(required) {
            failures.push(format!("required stage {required:?} is missing"));
        }
    }
    if policy.require_unique_producers {
        let producer_count = stages
            .iter()
            .map(|s| s.producer_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        if producer_count != stages.len() {
            failures.push("release stages do not have unique producers".to_string());
        }
    }
    let subject_fingerprint = subject_fingerprint.unwrap_or(0);
    if subject_fingerprint == 0 {
        failures.push("release corpus has no valid qualification subject".to_string());
    }
    HumanoidReleasePipelineReport {
        schema_version: HUMANOID_RELEASE_PIPELINE_SCHEMA_VERSION,
        release_id,
        policy_id: policy.policy_id.clone(),
        subject_fingerprint,
        evaluated_unix_millis: now_unix_millis,
        stages,
        passed: failures.is_empty(),
        failures,
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(stage: ReleaseStageKind, subject: u64, marker: char) -> ReleaseStageEvidence {
        ReleaseStageEvidence {
            stage,
            artifact_id: format!("artifact-{marker}"),
            artifact_sha256: marker.to_string().repeat(64),
            producer_id: format!("producer-{marker}"),
            passed: true,
            generated_unix_millis: 100,
            subject_fingerprint: subject,
        }
    }

    #[test]
    fn missing_stage_fails_closed() {
        let report = evaluate_release_pipeline(
            "r",
            &HumanoidReleasePipelinePolicy::default(),
            Vec::new(),
            100,
        );
        assert!(!report.passed);
        assert_eq!(report.subject_fingerprint, 0);
    }

    #[test]
    fn unbound_legacy_evidence_fails_closed() {
        let policy = HumanoidReleasePipelinePolicy {
            policy_id: "test".to_string(),
            required_stages: vec![ReleaseStageKind::SourceIntegrity],
            require_unique_producers: false,
            maximum_evidence_age_millis: 1_000,
        };
        let report = evaluate_release_pipeline(
            "r",
            &policy,
            vec![evidence(ReleaseStageKind::SourceIntegrity, 0, 'a')],
            100,
        );
        assert!(!report.passed);
    }

    #[test]
    fn mixed_subject_corpus_fails_closed() {
        let policy = HumanoidReleasePipelinePolicy {
            policy_id: "test".to_string(),
            required_stages: vec![
                ReleaseStageKind::SourceIntegrity,
                ReleaseStageKind::WorkspaceTests,
            ],
            require_unique_producers: false,
            maximum_evidence_age_millis: 1_000,
        };
        let report = evaluate_release_pipeline(
            "r",
            &policy,
            vec![
                evidence(ReleaseStageKind::SourceIntegrity, 7, 'a'),
                evidence(ReleaseStageKind::WorkspaceTests, 8, 'b'),
            ],
            100,
        );
        assert!(!report.passed);
        assert_eq!(report.subject_fingerprint, 7);
    }

    #[test]
    fn consistently_bound_minimal_policy_can_pass() {
        let policy = HumanoidReleasePipelinePolicy {
            policy_id: "test".to_string(),
            required_stages: vec![
                ReleaseStageKind::SourceIntegrity,
                ReleaseStageKind::WorkspaceTests,
            ],
            require_unique_producers: false,
            maximum_evidence_age_millis: 1_000,
        };
        let report = evaluate_release_pipeline(
            "r",
            &policy,
            vec![
                evidence(ReleaseStageKind::SourceIntegrity, 7, 'a'),
                evidence(ReleaseStageKind::WorkspaceTests, 7, 'b'),
            ],
            100,
        );
        assert!(report.passed, "{:?}", report.failures);
        assert_eq!(report.subject_fingerprint, 7);
    }
}
