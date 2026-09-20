// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Submission-set evidence for the BEHAVIOR-1K 2026 public-report protocol.
//!
//! This layer sits above the per-rollout import adapter. It distinguishes a
//! complete prescribed evaluation from a partial set and from an invalid set.
//! It does not prove that unreported external retries never occurred.

use crate::behavior_2026_observatory::{
    BEHAVIOR_1K_2026_REQUIRED_VERSION, Behavior2026EvaluationSplitV1,
    Behavior2026RolloutResultV1,
};
use crate::capability_observatory::CapabilitySubjectIdentityV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const BEHAVIOR_2026_SUBMISSION_SCHEMA_V1: &str =
    "symthaea.humanoid.behavior-2026-submission-set.v1";
pub const BEHAVIOR_2026_PUBLIC_TASK_COUNT: usize = 100;
pub const BEHAVIOR_2026_PUBLIC_INSTANCES_PER_TASK: usize = 10;
pub const BEHAVIOR_2026_PUBLIC_EXPECTED_ROLLOUTS: usize =
    BEHAVIOR_2026_PUBLIC_TASK_COUNT * BEHAVIOR_2026_PUBLIC_INSTANCES_PER_TASK;

const TASK_SET_DOMAIN_V1: &[u8] = b"symthaea:behavior-2026-public-task-set:v1\0";
const MANIFEST_DOMAIN_V1: &[u8] = b"symthaea:behavior-2026-public-manifest:v1\0";
const ROLLOUT_DOMAIN_V1: &[u8] = b"symthaea:behavior-2026-planned-rollout:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Behavior2026SubmissionManifestV1 {
    pub task_set_id: String,
    pub task_set_commitment: String,
    pub ordered_task_ids: Vec<String>,
    pub subject: CapabilitySubjectIdentityV1,
    pub wrapper_config_ref: String,
    pub robot_config_ref: String,
    pub manifest_commitment: String,
}

impl Behavior2026SubmissionManifestV1 {
    pub fn new(
        task_set_id: impl Into<String>,
        ordered_task_ids: Vec<String>,
        subject: CapabilitySubjectIdentityV1,
        wrapper_config_ref: impl Into<String>,
        robot_config_ref: impl Into<String>,
    ) -> Result<Self, Behavior2026SubmissionErrorV1> {
        let task_set_id = bounded_id(task_set_id.into())
            .ok_or(Behavior2026SubmissionErrorV1::InvalidTaskSetId)?;
        let wrapper_config_ref = bounded_id(wrapper_config_ref.into())
            .ok_or(Behavior2026SubmissionErrorV1::InvalidWrapperConfigRef)?;
        let robot_config_ref = bounded_id(robot_config_ref.into())
            .ok_or(Behavior2026SubmissionErrorV1::InvalidRobotConfigRef)?;

        if ordered_task_ids.len() != BEHAVIOR_2026_PUBLIC_TASK_COUNT {
            return Err(Behavior2026SubmissionErrorV1::WrongTaskCount);
        }
        let mut seen = BTreeSet::new();
        let mut canonical_tasks = Vec::with_capacity(ordered_task_ids.len());
        for task_id in ordered_task_ids {
            let task_id =
                bounded_id(task_id).ok_or(Behavior2026SubmissionErrorV1::InvalidTaskId)?;
            if !seen.insert(task_id.clone()) {
                return Err(Behavior2026SubmissionErrorV1::DuplicateTaskId);
            }
            canonical_tasks.push(task_id);
        }
        for value in [
            &subject.source_head,
            &subject.model_or_policy_id,
            &subject.morphology_id,
            &subject.sensor_actuator_profile_id,
        ] {
            if bounded_id(value.clone()).is_none() {
                return Err(Behavior2026SubmissionErrorV1::InvalidSubjectIdentity);
            }
        }

        let task_set_commitment = compute_task_set_commitment(&task_set_id, &canonical_tasks);
        let mut manifest = Self {
            task_set_id,
            task_set_commitment,
            ordered_task_ids: canonical_tasks,
            subject,
            wrapper_config_ref,
            robot_config_ref,
            manifest_commitment: String::new(),
        };
        manifest.manifest_commitment = manifest.compute_manifest_commitment();
        Ok(manifest)
    }

    fn compute_manifest_commitment(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(MANIFEST_DOMAIN_V1);
        hash_str(&mut hasher, &self.task_set_commitment);
        hash_str(&mut hasher, &self.subject.source_head);
        hash_str(&mut hasher, &self.subject.model_or_policy_id);
        hash_str(&mut hasher, &self.subject.morphology_id);
        hash_str(&mut hasher, &self.subject.sensor_actuator_profile_id);
        hash_str(&mut hasher, &self.wrapper_config_ref);
        hash_str(&mut hasher, &self.robot_config_ref);
        format!("behavior-2026-public:{}", hasher.finalize().to_hex())
    }

    pub fn planned_rollout_id(&self, task_id: &str, instance_index: u16) -> Option<String> {
        if instance_index > 9 || !self.ordered_task_ids.iter().any(|task| task == task_id) {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(ROLLOUT_DOMAIN_V1);
        hash_str(&mut hasher, &self.manifest_commitment);
        hash_str(&mut hasher, task_id);
        hasher.update(&instance_index.to_le_bytes());
        Some(format!("behavior-rollout:{}", hasher.finalize().to_hex()))
    }
}

fn compute_task_set_commitment(task_set_id: &str, ordered_task_ids: &[String]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(TASK_SET_DOMAIN_V1);
    hash_str(&mut hasher, BEHAVIOR_1K_2026_REQUIRED_VERSION);
    hash_str(&mut hasher, "public-report");
    hash_str(&mut hasher, task_set_id);
    hasher.update(&(ordered_task_ids.len() as u32).to_le_bytes());
    for (index, task_id) in ordered_task_ids.iter().enumerate() {
        hasher.update(&(index as u32).to_le_bytes());
        hash_str(&mut hasher, task_id);
    }
    format!("behavior-2026-task-set:{}", hasher.finalize().to_hex())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Behavior2026SubmissionDispositionV1 {
    CompletePrescribedSet,
    PartialPrescribedSet,
    InvalidSet,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Behavior2026SubmissionViolationV1 {
    MalformedRollout,
    WrongBenchmarkVersion,
    WrongEvaluationSplit,
    TaskSetMismatch,
    UnexpectedTask,
    WrapperConfigMismatch,
    RobotConfigMismatch,
    RolloutPlanMismatch,
    DuplicatePrescribedEntry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Behavior2026SubmissionReportV1 {
    pub task_set_commitment: String,
    pub manifest_commitment: String,
    pub disposition: Behavior2026SubmissionDispositionV1,
    pub expected_rollouts: usize,
    pub received_unique_prescribed_rollouts: usize,
    pub missing_rollouts: usize,
    pub coverage_rate: f64,
    /// Official-rule reconstruction for the prescribed public set. Missing
    /// entries contribute zero. `None` means the set itself is invalid.
    pub aggregate_q_score: Option<f64>,
    pub violations: Vec<Behavior2026SubmissionViolationV1>,
}

pub fn evaluate_behavior_2026_submission_v1(
    manifest: &Behavior2026SubmissionManifestV1,
    rollouts: &[Behavior2026RolloutResultV1],
) -> Behavior2026SubmissionReportV1 {
    let expected: BTreeSet<(String, u16)> = manifest
        .ordered_task_ids
        .iter()
        .flat_map(|task_id| (0u16..10).map(move |instance| (task_id.clone(), instance)))
        .collect();

    let mut accepted = BTreeMap::<(String, u16), f64>::new();
    let mut violations = BTreeSet::new();

    for rollout in rollouts {
        // Preserve identity diagnostics before generic structural validation.
        if rollout.benchmark_version != BEHAVIOR_1K_2026_REQUIRED_VERSION {
            violations.insert(Behavior2026SubmissionViolationV1::WrongBenchmarkVersion);
            continue;
        }
        if rollout.validate().is_err() {
            violations.insert(Behavior2026SubmissionViolationV1::MalformedRollout);
            continue;
        }
        if rollout.split != Behavior2026EvaluationSplitV1::PublicReport {
            violations.insert(Behavior2026SubmissionViolationV1::WrongEvaluationSplit);
            continue;
        }
        if rollout.task_set_id != manifest.task_set_id {
            violations.insert(Behavior2026SubmissionViolationV1::TaskSetMismatch);
            continue;
        }
        if rollout.wrapper_config_ref != manifest.wrapper_config_ref {
            violations.insert(Behavior2026SubmissionViolationV1::WrapperConfigMismatch);
            continue;
        }
        if rollout.robot_config_ref != manifest.robot_config_ref {
            violations.insert(Behavior2026SubmissionViolationV1::RobotConfigMismatch);
            continue;
        }

        let key = (rollout.task_id.clone(), rollout.instance_index);
        if !expected.contains(&key) {
            violations.insert(Behavior2026SubmissionViolationV1::UnexpectedTask);
            continue;
        }
        let planned = manifest
            .planned_rollout_id(&rollout.task_id, rollout.instance_index)
            .expect("expected key must have a planned rollout identity");
        if rollout.rollout_id != planned {
            violations.insert(Behavior2026SubmissionViolationV1::RolloutPlanMismatch);
            continue;
        }
        if accepted.insert(key, rollout.q_score).is_some() {
            violations.insert(Behavior2026SubmissionViolationV1::DuplicatePrescribedEntry);
        }
    }

    let received = accepted.len();
    let missing = BEHAVIOR_2026_PUBLIC_EXPECTED_ROLLOUTS.saturating_sub(received);
    let coverage_rate = received as f64 / BEHAVIOR_2026_PUBLIC_EXPECTED_ROLLOUTS as f64;
    let invalid = !violations.is_empty();
    let disposition = if invalid {
        Behavior2026SubmissionDispositionV1::InvalidSet
    } else if missing == 0 {
        Behavior2026SubmissionDispositionV1::CompletePrescribedSet
    } else {
        Behavior2026SubmissionDispositionV1::PartialPrescribedSet
    };
    let aggregate_q_score = if invalid {
        None
    } else {
        Some(
            accepted.values().copied().sum::<f64>()
                / BEHAVIOR_2026_PUBLIC_EXPECTED_ROLLOUTS as f64,
        )
    };

    Behavior2026SubmissionReportV1 {
        task_set_commitment: manifest.task_set_commitment.clone(),
        manifest_commitment: manifest.manifest_commitment.clone(),
        disposition,
        expected_rollouts: BEHAVIOR_2026_PUBLIC_EXPECTED_ROLLOUTS,
        received_unique_prescribed_rollouts: received,
        missing_rollouts: missing,
        coverage_rate,
        aggregate_q_score,
        violations: violations.into_iter().collect(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Behavior2026SubmissionErrorV1 {
    InvalidTaskSetId,
    InvalidTaskId,
    WrongTaskCount,
    DuplicateTaskId,
    InvalidSubjectIdentity,
    InvalidWrapperConfigRef,
    InvalidRobotConfigRef,
}

fn bounded_id(value: String) -> Option<String> {
    let value = value.trim().to_owned();
    (!value.is_empty() && value.len() <= 192).then_some(value)
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior_2026_observatory::{
        Behavior2026NativeMetricV1, Behavior2026PolicyInputProfileV1,
    };

    fn tasks() -> Vec<String> {
        (0..BEHAVIOR_2026_PUBLIC_TASK_COUNT)
            .map(|index| format!("task-{index:03}"))
            .collect()
    }

    fn subject() -> CapabilitySubjectIdentityV1 {
        CapabilitySubjectIdentityV1 {
            source_head: "candidate-head".into(),
            model_or_policy_id: "household-policy-v1".into(),
            morphology_id: "humanoid-v1".into(),
            sensor_actuator_profile_id: "behavior-sim-v1".into(),
        }
    }

    fn manifest() -> Behavior2026SubmissionManifestV1 {
        Behavior2026SubmissionManifestV1::new(
            "b100-v3.9.2",
            tasks(),
            subject(),
            "sha256:wrapper",
            "sha256:robot",
        )
        .unwrap()
    }

    fn rollout(
        manifest: &Behavior2026SubmissionManifestV1,
        task_id: &str,
        instance_index: u16,
        q_score: f64,
    ) -> Behavior2026RolloutResultV1 {
        Behavior2026RolloutResultV1 {
            benchmark_version: BEHAVIOR_1K_2026_REQUIRED_VERSION.into(),
            task_set_id: manifest.task_set_id.clone(),
            task_id: task_id.into(),
            instance_index,
            split: Behavior2026EvaluationSplitV1::PublicReport,
            rollout_id: manifest.planned_rollout_id(task_id, instance_index).unwrap(),
            scene_id: "scene-1".into(),
            policy_input_profile: Behavior2026PolicyInputProfileV1::RgbDepthProprioception,
            q_score,
            native_metrics: Vec::<Behavior2026NativeMetricV1>::new(),
            wrapper_config_ref: manifest.wrapper_config_ref.clone(),
            robot_config_ref: manifest.robot_config_ref.clone(),
            evaluator_artifact_ref: format!("artifact:{task_id}:{instance_index}"),
        }
    }

    #[test]
    fn manifest_requires_exactly_one_hundred_unique_tasks() {
        let mut too_short = tasks();
        too_short.pop();
        assert_eq!(
            Behavior2026SubmissionManifestV1::new(
                "set",
                too_short,
                subject(),
                "wrapper",
                "robot"
            ),
            Err(Behavior2026SubmissionErrorV1::WrongTaskCount)
        );

        let mut duplicate = tasks();
        duplicate[99] = duplicate[0].clone();
        assert_eq!(
            Behavior2026SubmissionManifestV1::new(
                "set",
                duplicate,
                subject(),
                "wrapper",
                "robot"
            ),
            Err(Behavior2026SubmissionErrorV1::DuplicateTaskId)
        );
    }

    #[test]
    fn task_set_and_manifest_commitments_are_deterministic_but_distinct() {
        let a = manifest();
        let b = manifest();
        assert_eq!(a.task_set_commitment, b.task_set_commitment);
        assert_eq!(a.manifest_commitment, b.manifest_commitment);
        assert_ne!(a.task_set_commitment, a.manifest_commitment);
        assert_eq!(
            a.planned_rollout_id("task-007", 3),
            b.planned_rollout_id("task-007", 3)
        );
    }

    #[test]
    fn task_order_changes_task_set_commitment() {
        let canonical = manifest();
        let mut reordered = tasks();
        reordered.swap(0, 1);
        let reordered = Behavior2026SubmissionManifestV1::new(
            "b100-v3.9.2",
            reordered,
            subject(),
            "sha256:wrapper",
            "sha256:robot",
        )
        .unwrap();
        assert_ne!(canonical.task_set_commitment, reordered.task_set_commitment);
    }

    #[test]
    fn complete_prescribed_set_is_distinct_from_partial() {
        let manifest = manifest();
        let mut rollouts = Vec::with_capacity(BEHAVIOR_2026_PUBLIC_EXPECTED_ROLLOUTS);
        for task_id in &manifest.ordered_task_ids {
            for instance in 0..10 {
                rollouts.push(rollout(&manifest, task_id, instance, 1.0));
            }
        }
        let report = evaluate_behavior_2026_submission_v1(&manifest, &rollouts);
        assert_eq!(
            report.disposition,
            Behavior2026SubmissionDispositionV1::CompletePrescribedSet
        );
        assert_eq!(report.task_set_commitment, manifest.task_set_commitment);
        assert_eq!(report.received_unique_prescribed_rollouts, 1000);
        assert_eq!(report.missing_rollouts, 0);
        assert_eq!(report.aggregate_q_score, Some(1.0));
    }

    #[test]
    fn missing_rollouts_contribute_zero_and_coverage_stays_visible() {
        let manifest = manifest();
        let one = rollout(&manifest, "task-000", 0, 1.0);
        let report = evaluate_behavior_2026_submission_v1(&manifest, &[one]);
        assert_eq!(
            report.disposition,
            Behavior2026SubmissionDispositionV1::PartialPrescribedSet
        );
        assert_eq!(report.received_unique_prescribed_rollouts, 1);
        assert_eq!(report.missing_rollouts, 999);
        assert!((report.coverage_rate - 0.001).abs() < f64::EPSILON);
        assert!((report.aggregate_q_score.unwrap() - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn duplicate_prescribed_entry_invalidates_set() {
        let manifest = manifest();
        let one = rollout(&manifest, "task-000", 0, 1.0);
        let report = evaluate_behavior_2026_submission_v1(&manifest, &[one.clone(), one]);
        assert_eq!(
            report.disposition,
            Behavior2026SubmissionDispositionV1::InvalidSet
        );
        assert!(report.aggregate_q_score.is_none());
        assert!(
            report
                .violations
                .contains(&Behavior2026SubmissionViolationV1::DuplicatePrescribedEntry)
        );
    }

    #[test]
    fn nonplanned_rollout_identity_invalidates_set() {
        let manifest = manifest();
        let mut one = rollout(&manifest, "task-000", 0, 1.0);
        one.rollout_id = "best-of-retries".into();
        let report = evaluate_behavior_2026_submission_v1(&manifest, &[one]);
        assert_eq!(
            report.disposition,
            Behavior2026SubmissionDispositionV1::InvalidSet
        );
        assert!(
            report
                .violations
                .contains(&Behavior2026SubmissionViolationV1::RolloutPlanMismatch)
        );
    }

    #[test]
    fn wrong_version_is_reported_specifically_before_generic_validation() {
        let manifest = manifest();
        let mut one = rollout(&manifest, "task-000", 0, 1.0);
        one.benchmark_version = "v3.9.1".into();
        let report = evaluate_behavior_2026_submission_v1(&manifest, &[one]);
        assert_eq!(
            report.disposition,
            Behavior2026SubmissionDispositionV1::InvalidSet
        );
        assert_eq!(
            report.violations,
            vec![Behavior2026SubmissionViolationV1::WrongBenchmarkVersion]
        );
    }

    #[test]
    fn config_mismatch_invalidates_set() {
        let manifest = manifest();
        let mut one = rollout(&manifest, "task-000", 0, 1.0);
        one.robot_config_ref = "different-robot-config".into();
        let report = evaluate_behavior_2026_submission_v1(&manifest, &[one]);
        assert_eq!(
            report.disposition,
            Behavior2026SubmissionDispositionV1::InvalidSet
        );
        assert!(
            report
                .violations
                .contains(&Behavior2026SubmissionViolationV1::RobotConfigMismatch)
        );
    }
}
