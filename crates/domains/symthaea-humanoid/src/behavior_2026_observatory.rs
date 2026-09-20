// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Claim-preserving import adapter for BEHAVIOR-1K 2026 Challenge results.
//!
//! This module imports already-produced external benchmark results. It does not
//! run OmniGibson, certify an official submission, or convert simulation into
//! physical-hardware evidence.

use crate::capability_observatory::{
    BenchmarkIdentityV1, CapabilityDomain, CapabilityMeasurementV1,
    CapabilityObservationError, CapabilitySubjectIdentityV1, ExecutionSubstrate, FailureEventV1,
    HumanoidCapabilityObservationV1, MeasurementProvenance, RunDisposition,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const BEHAVIOR_1K_BENCHMARK_ID: &str = "behavior-1k";
pub const BEHAVIOR_1K_2026_REQUIRED_VERSION: &str = "v3.9.2";
pub const BEHAVIOR_1K_2026_ADAPTER_SCHEMA_V1: &str =
    "symthaea.humanoid.behavior-1k-2026-result.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Behavior2026EvaluationSplitV1 {
    /// Challenge-reporting public instances 0..=9.
    PublicReport,
    /// Public development/test instances 10..=19.
    PublicDevelopment,
    /// Organizer-reserved final-evaluation instances 20..=39.
    HiddenFinal,
}

impl Behavior2026EvaluationSplitV1 {
    fn accepts(self, instance_index: u16) -> bool {
        match self {
            Self::PublicReport => instance_index <= 9,
            Self::PublicDevelopment => (10..=19).contains(&instance_index),
            Self::HiddenFinal => (20..=39).contains(&instance_index),
        }
    }

    fn as_id(self) -> &'static str {
        match self {
            Self::PublicReport => "public-report",
            Self::PublicDevelopment => "public-development",
            Self::HiddenFinal => "hidden-final",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Behavior2026PolicyInputProfileV1 {
    /// Official 2026 challenge-track policy inputs.
    RgbDepthProprioception,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Behavior2026NativeMetricV1 {
    pub metric_id: String,
    pub value: f64,
    pub unit: String,
}

impl Behavior2026NativeMetricV1 {
    fn validate(&self) -> Result<(), Behavior2026ObservationErrorV1> {
        validate_id(&self.metric_id, Behavior2026ObservationErrorV1::InvalidMetricId)?;
        if self.metric_id == "q_score" {
            return Err(Behavior2026ObservationErrorV1::ReservedMetricId);
        }
        if !self.value.is_finite() {
            return Err(Behavior2026ObservationErrorV1::NonFiniteMetric);
        }
        if self.unit.trim().is_empty() || self.unit.len() > 64 {
            return Err(Behavior2026ObservationErrorV1::InvalidMetricUnit);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Behavior2026RolloutResultV1 {
    pub benchmark_version: String,
    pub task_set_id: String,
    pub task_id: String,
    pub instance_index: u16,
    pub split: Behavior2026EvaluationSplitV1,
    pub rollout_id: String,
    pub scene_id: String,
    pub policy_input_profile: Behavior2026PolicyInputProfileV1,
    pub q_score: f64,
    pub native_metrics: Vec<Behavior2026NativeMetricV1>,
    pub wrapper_config_ref: String,
    pub robot_config_ref: String,
    pub evaluator_artifact_ref: String,
}

impl Behavior2026RolloutResultV1 {
    pub fn validate(&self) -> Result<(), Behavior2026ObservationErrorV1> {
        if self.benchmark_version != BEHAVIOR_1K_2026_REQUIRED_VERSION {
            return Err(Behavior2026ObservationErrorV1::WrongBenchmarkVersion);
        }
        validate_id(
            &self.task_set_id,
            Behavior2026ObservationErrorV1::InvalidTaskSetId,
        )?;
        validate_id(&self.task_id, Behavior2026ObservationErrorV1::InvalidTaskId)?;
        validate_id(
            &self.rollout_id,
            Behavior2026ObservationErrorV1::InvalidRolloutId,
        )?;
        validate_id(&self.scene_id, Behavior2026ObservationErrorV1::InvalidSceneId)?;
        validate_id(
            &self.wrapper_config_ref,
            Behavior2026ObservationErrorV1::InvalidWrapperConfigRef,
        )?;
        validate_id(
            &self.robot_config_ref,
            Behavior2026ObservationErrorV1::InvalidRobotConfigRef,
        )?;
        validate_id(
            &self.evaluator_artifact_ref,
            Behavior2026ObservationErrorV1::InvalidEvaluatorArtifactRef,
        )?;
        if !self.split.accepts(self.instance_index) {
            return Err(Behavior2026ObservationErrorV1::InstanceSplitMismatch);
        }
        if !self.q_score.is_finite() || !(0.0..=1.0).contains(&self.q_score) {
            return Err(Behavior2026ObservationErrorV1::InvalidQScore);
        }

        let mut metric_ids = BTreeSet::new();
        for metric in &self.native_metrics {
            metric.validate()?;
            if !metric_ids.insert(metric.metric_id.as_str()) {
                return Err(Behavior2026ObservationErrorV1::DuplicateMetricId);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Behavior2026ObservationContextV1 {
    pub subject: CapabilitySubjectIdentityV1,
    pub authority_profile_id: String,
    pub evidence_profile_id: String,
}

impl Behavior2026ObservationContextV1 {
    fn validate(&self) -> Result<(), Behavior2026ObservationErrorV1> {
        validate_id(
            &self.authority_profile_id,
            Behavior2026ObservationErrorV1::InvalidAuthorityProfileId,
        )?;
        validate_id(
            &self.evidence_profile_id,
            Behavior2026ObservationErrorV1::InvalidEvidenceProfileId,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Behavior2026ObservationErrorV1 {
    WrongBenchmarkVersion,
    InvalidTaskSetId,
    InvalidTaskId,
    InvalidRolloutId,
    InvalidSceneId,
    InvalidWrapperConfigRef,
    InvalidRobotConfigRef,
    InvalidEvaluatorArtifactRef,
    InvalidAuthorityProfileId,
    InvalidEvidenceProfileId,
    InstanceSplitMismatch,
    InvalidQScore,
    InvalidMetricId,
    ReservedMetricId,
    InvalidMetricUnit,
    NonFiniteMetric,
    DuplicateMetricId,
    CapabilityObservation(CapabilityObservationError),
}

impl From<CapabilityObservationError> for Behavior2026ObservationErrorV1 {
    fn from(value: CapabilityObservationError) -> Self {
        Self::CapabilityObservation(value)
    }
}

pub fn import_behavior_2026_rollout_v1(
    result: &Behavior2026RolloutResultV1,
    context: &Behavior2026ObservationContextV1,
) -> Result<HumanoidCapabilityObservationV1, Behavior2026ObservationErrorV1> {
    result.validate()?;
    context.validate()?;

    let mut measurements = vec![CapabilityMeasurementV1 {
        metric_id: "behavior.q_score".to_owned(),
        value: result.q_score,
        unit: "ratio".to_owned(),
        provenance: MeasurementProvenance::BenchmarkNative,
    }];
    measurements.extend(result.native_metrics.iter().map(|metric| CapabilityMeasurementV1 {
        metric_id: format!("behavior.{}", metric.metric_id),
        value: metric.value,
        unit: metric.unit.clone(),
        provenance: MeasurementProvenance::BenchmarkNative,
    }));

    let mut failures = Vec::new();
    if result.q_score < 1.0 {
        failures.push(FailureEventV1 {
            failure_id: "behavior-task-incomplete".to_owned(),
            category: "task-incomplete".to_owned(),
            description: format!(
                "BEHAVIOR task did not satisfy every goal predicate; q_score={:.6}",
                result.q_score
            ),
        });
    }

    let run_id = format!(
        "behavior-2026:{}:{}:{}:{}",
        result.task_id,
        result.split.as_id(),
        result.instance_index,
        result.rollout_id
    );
    let environment_id = format!("behavior-2026:{}:{}", result.scene_id, result.split.as_id());

    Ok(HumanoidCapabilityObservationV1::new(
        HumanoidCapabilityObservationV1 {
            run_id,
            domain: CapabilityDomain::HouseholdTask,
            substrate: ExecutionSubstrate::ExternalBenchmarkSimulation,
            subject: context.subject.clone(),
            benchmark: BenchmarkIdentityV1::External {
                benchmark_id: BEHAVIOR_1K_BENCHMARK_ID.to_owned(),
                benchmark_version: BEHAVIOR_1K_2026_REQUIRED_VERSION.to_owned(),
                task_set_id: result.task_set_id.clone(),
            },
            environment_id,
            authority_profile_id: context.authority_profile_id.clone(),
            evidence_profile_id: context.evidence_profile_id.clone(),
            disposition: RunDisposition::Completed,
            measurements,
            failures,
        },
    )?)
}

fn validate_id(value: &str, error: Behavior2026ObservationErrorV1) -> Result<(), Behavior2026ObservationErrorV1> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 192 {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> Behavior2026ObservationContextV1 {
        Behavior2026ObservationContextV1 {
            subject: CapabilitySubjectIdentityV1 {
                source_head: "deadbeef".into(),
                model_or_policy_id: "household-policy-v1".into(),
                morphology_id: "humanoid-v1".into(),
                sensor_actuator_profile_id: "behavior-sim-v1".into(),
            },
            authority_profile_id: "simulation-no-human-authority".into(),
            evidence_profile_id: "external-benchmark-import-v1".into(),
        }
    }

    fn result(q_score: f64) -> Behavior2026RolloutResultV1 {
        Behavior2026RolloutResultV1 {
            benchmark_version: BEHAVIOR_1K_2026_REQUIRED_VERSION.into(),
            task_set_id: "2026-challenge-b100".into(),
            task_id: "turning_on_radio".into(),
            instance_index: 3,
            split: Behavior2026EvaluationSplitV1::PublicReport,
            rollout_id: "rollout-0003".into(),
            scene_id: "living-room-1".into(),
            policy_input_profile: Behavior2026PolicyInputProfileV1::RgbDepthProprioception,
            q_score,
            native_metrics: vec![Behavior2026NativeMetricV1 {
                metric_id: "time".into(),
                value: 42.0,
                unit: "s".into(),
            }],
            wrapper_config_ref: "sha256:wrapper".into(),
            robot_config_ref: "sha256:robot".into(),
            evaluator_artifact_ref: "sha256:result-json".into(),
        }
    }

    #[test]
    fn valid_result_imports_as_external_household_observation() {
        let observation = import_behavior_2026_rollout_v1(&result(1.0), &context()).unwrap();
        assert_eq!(observation.domain, CapabilityDomain::HouseholdTask);
        assert_eq!(
            observation.substrate,
            ExecutionSubstrate::ExternalBenchmarkSimulation
        );
        assert!(matches!(
            observation.benchmark,
            BenchmarkIdentityV1::External { ref benchmark_id, ref benchmark_version, .. }
                if benchmark_id == BEHAVIOR_1K_BENCHMARK_ID
                    && benchmark_version == BEHAVIOR_1K_2026_REQUIRED_VERSION
        ));
        assert!(observation.failures.is_empty());
    }

    #[test]
    fn partial_success_is_negative_capability_evidence_not_execution_failure() {
        let observation = import_behavior_2026_rollout_v1(&result(0.4), &context()).unwrap();
        assert_eq!(observation.disposition, RunDisposition::Completed);
        assert_eq!(observation.failures.len(), 1);
        assert_eq!(observation.failures[0].category, "task-incomplete");
    }

    #[test]
    fn wrong_benchmark_version_fails_closed() {
        let mut result = result(1.0);
        result.benchmark_version = "v3.9.0".into();
        assert_eq!(
            import_behavior_2026_rollout_v1(&result, &context()),
            Err(Behavior2026ObservationErrorV1::WrongBenchmarkVersion)
        );
    }

    #[test]
    fn invalid_q_score_fails_closed() {
        assert_eq!(
            import_behavior_2026_rollout_v1(&result(1.1), &context()),
            Err(Behavior2026ObservationErrorV1::InvalidQScore)
        );
    }

    #[test]
    fn evaluation_split_and_instance_population_must_match() {
        let mut result = result(1.0);
        result.instance_index = 17;
        assert_eq!(
            import_behavior_2026_rollout_v1(&result, &context()),
            Err(Behavior2026ObservationErrorV1::InstanceSplitMismatch)
        );
    }

    #[test]
    fn duplicate_native_metric_ids_fail_closed() {
        let mut result = result(1.0);
        result.native_metrics.push(result.native_metrics[0].clone());
        assert_eq!(
            import_behavior_2026_rollout_v1(&result, &context()),
            Err(Behavior2026ObservationErrorV1::DuplicateMetricId)
        );
    }

    #[test]
    fn q_score_cannot_be_smuggled_as_optional_duplicate_metric() {
        let mut result = result(1.0);
        result.native_metrics.push(Behavior2026NativeMetricV1 {
            metric_id: "q_score".into(),
            value: 0.0,
            unit: "ratio".into(),
        });
        assert_eq!(
            import_behavior_2026_rollout_v1(&result, &context()),
            Err(Behavior2026ObservationErrorV1::ReservedMetricId)
        );
    }
}
