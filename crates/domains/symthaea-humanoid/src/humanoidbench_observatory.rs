// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Claim-preserving HumanoidBench import adapter.
//!
//! This module imports metadata/results produced by an external HumanoidBench
//! runner. It does not embed or reimplement the upstream Python benchmark and
//! it does not promote simulation evidence into physical capability claims.

use crate::capability_observatory::{
    BenchmarkIdentityV1, CapabilityDomain, CapabilityMeasurementV1, CapabilitySubjectIdentityV1,
    ExecutionSubstrate, FailureEventV1, HumanoidCapabilityObservationV1, MeasurementProvenance,
    RunDisposition,
};
use serde::{Deserialize, Serialize};

pub const HUMANOIDBENCH_ADAPTER_SCHEMA_V1: &str =
    "symthaea.humanoid.humanoidbench-observation.v1";
pub const HUMANOIDBENCH_UPSTREAM_REPOSITORY: &str = "carlosferrazza/humanoid-bench";
pub const HUMANOIDBENCH_UPSTREAM_COMMIT: &str =
    "cb1189039151c8aadaaa987b442da54383c87fab";
const TASK_REGISTRY_DOMAIN_V1: &[u8] = b"symthaea:humanoidbench-task-registry:v1\0";

/// Exact `TASKS` keys in upstream `humanoid_bench/env.py` at
/// `HUMANOIDBENCH_UPSTREAM_COMMIT`, kept in canonical lexical order here.
pub const HUMANOIDBENCH_TASK_REGISTRY_V1: [&str; 32] = [
    "balance_hard",
    "balance_simple",
    "basketball",
    "bookshelf_hard",
    "bookshelf_simple",
    "cabinet",
    "crawl",
    "cube",
    "door",
    "highbar_hard",
    "highbar_simple",
    "hurdle",
    "insert_normal",
    "insert_small",
    "kitchen",
    "maze",
    "package",
    "pole",
    "powerlift",
    "push",
    "reach",
    "room",
    "run",
    "sit_hard",
    "sit_simple",
    "slide",
    "spoon",
    "stair",
    "stand",
    "truck",
    "walk",
    "window",
];

/// Exact `ROBOTS` keys in upstream `humanoid_bench/env.py` at the pinned commit.
/// Robot identity is per-run evidence and is intentionally not part of the
/// task-registry commitment.
pub const HUMANOIDBENCH_ROBOT_REGISTRY_V1: [&str; 6] = [
    "g1",
    "h1",
    "h1hand",
    "h1simplehand",
    "h1strong",
    "h1touch",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidBenchControlModeV1 {
    Position,
    Torque,
}

impl HumanoidBenchControlModeV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Position => "pos",
            Self::Torque => "torque",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidBenchExecutionStatusV1 {
    Completed,
    InfrastructureFailure,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanoidBenchEpisodeResultV1 {
    pub upstream_commit: String,
    pub task_id: String,
    pub robot_id: String,
    pub control_mode: HumanoidBenchControlModeV1,
    pub episode_id: String,
    /// `None` means the runner did not expose a reproducible seed. The exact
    /// integer remains adapter metadata and is not lossy-converted to f64.
    pub random_seed: Option<u64>,
    pub execution_status: HumanoidBenchExecutionStatusV1,
    /// Required for completed episodes; absent for an episode that did not
    /// execute far enough to establish benchmark-native return evidence.
    pub episode_return: Option<f64>,
    pub episode_length_steps: Option<u64>,
    pub terminated: Option<bool>,
    pub truncated: Option<bool>,
    /// Required only for infrastructure failure. This is operational evidence,
    /// not a negative robot-capability judgment.
    pub infrastructure_failure: Option<String>,
    /// Opaque reference to the original runner/evaluator artifact. The adapter
    /// record remains the lossless import evidence even though the common
    /// numeric observation envelope cannot embed arbitrary string metadata.
    pub runner_artifact_ref: String,
    pub environment_id: String,
    pub authority_profile_id: String,
    pub evidence_profile_id: String,
}

impl HumanoidBenchEpisodeResultV1 {
    pub fn validate(&self) -> Result<(), HumanoidBenchAdapterErrorV1> {
        if self.upstream_commit != HUMANOIDBENCH_UPSTREAM_COMMIT {
            return Err(HumanoidBenchAdapterErrorV1::WrongUpstreamCommit);
        }
        if !HUMANOIDBENCH_TASK_REGISTRY_V1.contains(&self.task_id.as_str()) {
            return Err(HumanoidBenchAdapterErrorV1::UnknownTask);
        }
        if !HUMANOIDBENCH_ROBOT_REGISTRY_V1.contains(&self.robot_id.as_str()) {
            return Err(HumanoidBenchAdapterErrorV1::UnknownRobot);
        }
        let expected_control = if self.robot_id == "g1" {
            HumanoidBenchControlModeV1::Torque
        } else {
            HumanoidBenchControlModeV1::Position
        };
        if self.control_mode != expected_control {
            return Err(HumanoidBenchAdapterErrorV1::ControlModeMismatch);
        }
        for value in [
            &self.episode_id,
            &self.runner_artifact_ref,
            &self.environment_id,
            &self.authority_profile_id,
            &self.evidence_profile_id,
        ] {
            if !valid_id(value) {
                return Err(HumanoidBenchAdapterErrorV1::InvalidIdentity);
            }
        }
        if self.episode_return.is_some_and(|value| !value.is_finite()) {
            return Err(HumanoidBenchAdapterErrorV1::NonFiniteReturn);
        }

        match self.execution_status {
            HumanoidBenchExecutionStatusV1::Completed => {
                if self.episode_return.is_none()
                    || self.episode_length_steps.is_none()
                    || self.terminated.is_none()
                    || self.truncated.is_none()
                {
                    return Err(HumanoidBenchAdapterErrorV1::CompletedMissingEvidence);
                }
                if self.infrastructure_failure.is_some() {
                    return Err(HumanoidBenchAdapterErrorV1::CompletedHasInfrastructureFailure);
                }
            }
            HumanoidBenchExecutionStatusV1::InfrastructureFailure => {
                let Some(reason) = self.infrastructure_failure.as_deref() else {
                    return Err(HumanoidBenchAdapterErrorV1::InfrastructureFailureMissingReason);
                };
                if reason.trim().is_empty() || reason.len() > 512 {
                    return Err(HumanoidBenchAdapterErrorV1::InvalidInfrastructureFailureReason);
                }
            }
        }
        Ok(())
    }

    pub fn to_observation(
        &self,
        subject: CapabilitySubjectIdentityV1,
    ) -> Result<HumanoidCapabilityObservationV1, HumanoidBenchAdapterErrorV1> {
        self.validate()?;

        let mut measurements = Vec::new();
        if let Some(value) = self.episode_return {
            measurements.push(CapabilityMeasurementV1 {
                metric_id: "humanoidbench.episode_return".into(),
                value,
                unit: "benchmark_return".into(),
                provenance: MeasurementProvenance::BenchmarkNative,
            });
        }
        if let Some(value) = self.episode_length_steps {
            measurements.push(CapabilityMeasurementV1 {
                metric_id: "humanoidbench.episode_length_steps".into(),
                value: value as f64,
                unit: "steps".into(),
                provenance: MeasurementProvenance::BenchmarkNative,
            });
        }
        if let Some(value) = self.terminated {
            measurements.push(CapabilityMeasurementV1 {
                metric_id: "humanoidbench.terminated".into(),
                value: if value { 1.0 } else { 0.0 },
                unit: "bool01".into(),
                provenance: MeasurementProvenance::BenchmarkNative,
            });
        }
        if let Some(value) = self.truncated {
            measurements.push(CapabilityMeasurementV1 {
                metric_id: "humanoidbench.truncated".into(),
                value: if value { 1.0 } else { 0.0 },
                unit: "bool01".into(),
                provenance: MeasurementProvenance::BenchmarkNative,
            });
        }

        let (disposition, failures) = match self.execution_status {
            HumanoidBenchExecutionStatusV1::Completed => (RunDisposition::Completed, Vec::new()),
            HumanoidBenchExecutionStatusV1::InfrastructureFailure => (
                RunDisposition::InfrastructureIndeterminate,
                vec![FailureEventV1 {
                    failure_id: format!("{}:infrastructure", self.episode_id),
                    category: "external_benchmark_infrastructure".into(),
                    description: self
                        .infrastructure_failure
                        .clone()
                        .expect("validated infrastructure failure has a reason"),
                }],
            ),
        };

        HumanoidCapabilityObservationV1::new(HumanoidCapabilityObservationV1 {
            run_id: self.episode_id.clone(),
            domain: task_domain(&self.task_id),
            substrate: ExecutionSubstrate::ExternalBenchmarkSimulation,
            subject,
            benchmark: BenchmarkIdentityV1::External {
                benchmark_id: HUMANOIDBENCH_UPSTREAM_REPOSITORY.into(),
                benchmark_version: HUMANOIDBENCH_UPSTREAM_COMMIT.into(),
                task_set_id: humanoidbench_task_registry_commitment_v1(),
            },
            environment_id: format!(
                "{}:{}:{}:{}",
                self.environment_id,
                self.robot_id,
                self.control_mode.as_str(),
                self.task_id
            ),
            authority_profile_id: self.authority_profile_id.clone(),
            evidence_profile_id: self.evidence_profile_id.clone(),
            disposition,
            measurements,
            failures,
        })
        .map_err(|_| HumanoidBenchAdapterErrorV1::InvalidObservatoryEnvelope)
    }
}

/// Commitment to the exact upstream task vocabulary at the pinned source commit.
/// Robot/control identity is intentionally excluded because it is per-run
/// evidence rather than part of the task-set definition.
pub fn humanoidbench_task_registry_commitment_v1() -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(TASK_REGISTRY_DOMAIN_V1);
    hash_str(&mut hasher, HUMANOIDBENCH_UPSTREAM_REPOSITORY);
    hash_str(&mut hasher, HUMANOIDBENCH_UPSTREAM_COMMIT);
    hasher.update(&(HUMANOIDBENCH_TASK_REGISTRY_V1.len() as u32).to_le_bytes());
    for task in HUMANOIDBENCH_TASK_REGISTRY_V1 {
        hash_str(&mut hasher, task);
    }
    format!("humanoidbench-tasks:{}", hasher.finalize().to_hex())
}

fn task_domain(task_id: &str) -> CapabilityDomain {
    match task_id {
        "balance_hard" | "balance_simple" => CapabilityDomain::BalanceRecovery,
        "stand" | "walk" | "run" | "maze" | "hurdle" | "crawl" | "highbar_hard"
        | "highbar_simple" | "sit_hard" | "sit_simple" | "stair" | "slide" | "pole" => {
            CapabilityDomain::Locomotion
        }
        _ => CapabilityDomain::MobileManipulation,
    }
}

fn valid_id(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty() && value.len() <= 192
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanoidBenchAdapterErrorV1 {
    WrongUpstreamCommit,
    UnknownTask,
    UnknownRobot,
    ControlModeMismatch,
    InvalidIdentity,
    NonFiniteReturn,
    CompletedMissingEvidence,
    CompletedHasInfrastructureFailure,
    InfrastructureFailureMissingReason,
    InvalidInfrastructureFailureReason,
    InvalidObservatoryEnvelope,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> CapabilitySubjectIdentityV1 {
        CapabilitySubjectIdentityV1 {
            source_head: "candidate-head".into(),
            model_or_policy_id: "policy-v1".into(),
            morphology_id: "humanoid-v1".into(),
            sensor_actuator_profile_id: "sim-v1".into(),
        }
    }

    fn completed() -> HumanoidBenchEpisodeResultV1 {
        HumanoidBenchEpisodeResultV1 {
            upstream_commit: HUMANOIDBENCH_UPSTREAM_COMMIT.into(),
            task_id: "walk".into(),
            robot_id: "h1hand".into(),
            control_mode: HumanoidBenchControlModeV1::Position,
            episode_id: "episode-1".into(),
            random_seed: Some(7),
            execution_status: HumanoidBenchExecutionStatusV1::Completed,
            episode_return: Some(-100.0),
            episode_length_steps: Some(1000),
            terminated: Some(false),
            truncated: Some(true),
            infrastructure_failure: None,
            runner_artifact_ref: "artifact:runner-1".into(),
            environment_id: "mujoco-upstream".into(),
            authority_profile_id: "simulation-no-human-authority".into(),
            evidence_profile_id: "external-import-only".into(),
        }
    }

    #[test]
    fn exact_task_registry_identity_is_deterministic() {
        let a = humanoidbench_task_registry_commitment_v1();
        let b = humanoidbench_task_registry_commitment_v1();
        assert_eq!(a, b);
        assert!(a.starts_with("humanoidbench-tasks:"));
    }

    #[test]
    fn task_set_identity_is_independent_of_robot_selection() {
        let a = humanoidbench_task_registry_commitment_v1();
        let mut result = completed();
        result.robot_id = "g1".into();
        result.control_mode = HumanoidBenchControlModeV1::Torque;
        assert!(result.validate().is_ok());
        assert_eq!(a, humanoidbench_task_registry_commitment_v1());
    }

    #[test]
    fn unknown_task_fails_closed() {
        let mut result = completed();
        result.task_id = "future-task".into();
        assert_eq!(result.validate(), Err(HumanoidBenchAdapterErrorV1::UnknownTask));
    }

    #[test]
    fn wrong_upstream_commit_fails_closed() {
        let mut result = completed();
        result.upstream_commit = "main".into();
        assert_eq!(
            result.validate(),
            Err(HumanoidBenchAdapterErrorV1::WrongUpstreamCommit)
        );
    }

    #[test]
    fn upstream_control_mode_is_bound_to_robot_registry() {
        let mut result = completed();
        result.robot_id = "g1".into();
        assert_eq!(
            result.validate(),
            Err(HumanoidBenchAdapterErrorV1::ControlModeMismatch)
        );
        result.control_mode = HumanoidBenchControlModeV1::Torque;
        assert!(result.validate().is_ok());
    }

    #[test]
    fn low_return_is_completed_execution_not_infrastructure_failure() {
        let observation = completed().to_observation(subject()).unwrap();
        assert_eq!(observation.disposition, RunDisposition::Completed);
        assert!(observation.failures.is_empty());
        assert_eq!(observation.domain, CapabilityDomain::Locomotion);
        assert!(observation.measurements.iter().any(|measurement| {
            measurement.metric_id == "humanoidbench.episode_return"
                && measurement.value == -100.0
        }));
    }

    #[test]
    fn exact_seed_stays_adapter_metadata_not_lossy_measurement() {
        let observation = completed().to_observation(subject()).unwrap();
        assert!(observation
            .measurements
            .iter()
            .all(|measurement| measurement.metric_id != "humanoidbench.random_seed"));
    }

    #[test]
    fn infrastructure_failure_remains_indeterminate() {
        let mut result = completed();
        result.execution_status = HumanoidBenchExecutionStatusV1::InfrastructureFailure;
        result.episode_return = None;
        result.episode_length_steps = None;
        result.terminated = None;
        result.truncated = None;
        result.infrastructure_failure = Some("runner process exited before episode start".into());
        let observation = result.to_observation(subject()).unwrap();
        assert_eq!(observation.disposition, RunDisposition::InfrastructureIndeterminate);
        assert_eq!(observation.failures.len(), 1);
    }

    #[test]
    fn non_finite_return_fails_closed() {
        let mut result = completed();
        result.episode_return = Some(f64::NAN);
        assert_eq!(
            result.validate(),
            Err(HumanoidBenchAdapterErrorV1::NonFiniteReturn)
        );
    }
}
