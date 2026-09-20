// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lossless exact-metadata lowering for the pinned HumanoidBench adapter.

use crate::capability_observatory::{CapabilitySubjectIdentityV1, MeasurementProvenance};
use crate::capability_observatory_v2::{
    CapabilityExactMetadataV1, CapabilityExactMetadataValueV1, HumanoidCapabilityObservationV2,
};
use crate::humanoidbench_observatory::{
    HumanoidBenchControlModeV1, HumanoidBenchEpisodeResultV1, HumanoidBenchExecutionStatusV1,
};

pub const HUMANOIDBENCH_TYPED_METADATA_SCHEMA_V1: &str =
    "symthaea.humanoid.humanoidbench-typed-metadata.v1";

impl HumanoidBenchEpisodeResultV1 {
    /// Compatibility-preserving V2 lowering.
    ///
    /// The existing V1 observation is retained unchanged. Exact integer,
    /// boolean, token, and reference evidence is carried alongside it without
    /// float coercion or compound-identifier parsing.
    pub fn to_observation_v2(
        &self,
        subject: CapabilitySubjectIdentityV1,
    ) -> Result<HumanoidCapabilityObservationV2, HumanoidBenchAdapterV2Error> {
        let observation = self
            .to_observation(subject)
            .map_err(|_| HumanoidBenchAdapterV2Error::InvalidSourceEvidence)?;

        let mut exact_metadata = vec![
            benchmark_token("humanoidbench.task_id", self.task_id.clone()),
            benchmark_token("humanoidbench.robot_id", self.robot_id.clone()),
            benchmark_token(
                "humanoidbench.control_mode",
                match self.control_mode {
                    HumanoidBenchControlModeV1::Position => "position",
                    HumanoidBenchControlModeV1::Torque => "torque",
                },
            ),
            benchmark_token(
                "humanoidbench.execution_status",
                match self.execution_status {
                    HumanoidBenchExecutionStatusV1::Completed => "completed",
                    HumanoidBenchExecutionStatusV1::InfrastructureFailure => {
                        "infrastructure_failure"
                    }
                },
            ),
            benchmark_reference(
                "humanoidbench.runner_artifact_ref",
                self.runner_artifact_ref.clone(),
                Some("symthaea.external-runner-artifact-ref.v1"),
            ),
            benchmark_reference(
                "humanoidbench.upstream_commit",
                self.upstream_commit.clone(),
                Some("git.commit-sha1"),
            ),
        ];

        match self.random_seed {
            Some(seed) => {
                exact_metadata.push(benchmark_token(
                    "humanoidbench.seed_state",
                    "specified",
                ));
                exact_metadata.push(benchmark_unsigned(
                    "humanoidbench.random_seed",
                    seed,
                    None,
                ));
            }
            None => exact_metadata.push(benchmark_token(
                "humanoidbench.seed_state",
                "unspecified",
            )),
        }

        if let Some(length) = self.episode_length_steps {
            exact_metadata.push(benchmark_unsigned(
                "humanoidbench.episode_length_steps_exact",
                length,
                Some("steps"),
            ));
        }
        if let Some(terminated) = self.terminated {
            exact_metadata.push(benchmark_bool("humanoidbench.terminated", terminated));
        }
        if let Some(truncated) = self.truncated {
            exact_metadata.push(benchmark_bool("humanoidbench.truncated", truncated));
        }

        HumanoidCapabilityObservationV2::new(observation, exact_metadata)
            .map_err(|_| HumanoidBenchAdapterV2Error::InvalidTypedEnvelope)
    }
}

fn benchmark_token(
    field_id: &str,
    value: impl Into<String>,
) -> CapabilityExactMetadataV1 {
    CapabilityExactMetadataV1 {
        field_id: field_id.into(),
        value: CapabilityExactMetadataValueV1::Token(value.into()),
        provenance: MeasurementProvenance::BenchmarkNative,
        unit: None,
        schema_id: None,
    }
}

fn benchmark_reference(
    field_id: &str,
    value: impl Into<String>,
    schema_id: Option<&str>,
) -> CapabilityExactMetadataV1 {
    CapabilityExactMetadataV1 {
        field_id: field_id.into(),
        value: CapabilityExactMetadataValueV1::Reference(value.into()),
        provenance: MeasurementProvenance::BenchmarkNative,
        unit: None,
        schema_id: schema_id.map(str::to_owned),
    }
}

fn benchmark_unsigned(
    field_id: &str,
    value: u64,
    unit: Option<&str>,
) -> CapabilityExactMetadataV1 {
    CapabilityExactMetadataV1 {
        field_id: field_id.into(),
        value: CapabilityExactMetadataValueV1::Unsigned(value),
        provenance: MeasurementProvenance::BenchmarkNative,
        unit: unit.map(str::to_owned),
        schema_id: None,
    }
}

fn benchmark_bool(field_id: &str, value: bool) -> CapabilityExactMetadataV1 {
    CapabilityExactMetadataV1 {
        field_id: field_id.into(),
        value: CapabilityExactMetadataValueV1::Bool(value),
        provenance: MeasurementProvenance::BenchmarkNative,
        unit: None,
        schema_id: None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanoidBenchAdapterV2Error {
    InvalidSourceEvidence,
    InvalidTypedEnvelope,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::humanoidbench_observatory::{
        HUMANOIDBENCH_UPSTREAM_COMMIT, HumanoidBenchAdapterErrorV1,
    };

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
            episode_id: "typed-episode-1".into(),
            random_seed: Some(u64::MAX),
            execution_status: HumanoidBenchExecutionStatusV1::Completed,
            episode_return: Some(5.0),
            episode_length_steps: Some(u64::MAX),
            terminated: Some(false),
            truncated: Some(true),
            infrastructure_failure: None,
            runner_artifact_ref: "artifact:runner-exact".into(),
            environment_id: "mujoco-upstream".into(),
            authority_profile_id: "simulation-no-human-authority".into(),
            evidence_profile_id: "external-import-only".into(),
        }
    }

    #[test]
    fn exact_seed_and_step_count_survive_as_unsigned_metadata() {
        let observation = completed().to_observation_v2(subject()).unwrap();
        assert_eq!(
            observation
                .metadata("humanoidbench.random_seed")
                .map(|item| &item.value),
            Some(&CapabilityExactMetadataValueV1::Unsigned(u64::MAX))
        );
        assert_eq!(
            observation
                .metadata("humanoidbench.episode_length_steps_exact")
                .map(|item| &item.value),
            Some(&CapabilityExactMetadataValueV1::Unsigned(u64::MAX))
        );
    }

    #[test]
    fn terminated_and_truncated_are_typed_booleans() {
        let observation = completed().to_observation_v2(subject()).unwrap();
        assert_eq!(
            observation
                .metadata("humanoidbench.terminated")
                .map(|item| &item.value),
            Some(&CapabilityExactMetadataValueV1::Bool(false))
        );
        assert_eq!(
            observation
                .metadata("humanoidbench.truncated")
                .map(|item| &item.value),
            Some(&CapabilityExactMetadataValueV1::Bool(true))
        );
    }

    #[test]
    fn unspecified_seed_is_explicit_state_not_fake_zero() {
        let mut result = completed();
        result.random_seed = None;
        let observation = result.to_observation_v2(subject()).unwrap();
        assert!(observation.metadata("humanoidbench.random_seed").is_none());
        assert_eq!(
            observation
                .metadata("humanoidbench.seed_state")
                .map(|item| &item.value),
            Some(&CapabilityExactMetadataValueV1::Token("unspecified".into()))
        );
    }

    #[test]
    fn exact_runner_reference_is_preserved() {
        let observation = completed().to_observation_v2(subject()).unwrap();
        assert_eq!(
            observation
                .metadata("humanoidbench.runner_artifact_ref")
                .map(|item| &item.value),
            Some(&CapabilityExactMetadataValueV1::Reference(
                "artifact:runner-exact".into()
            ))
        );
    }

    #[test]
    fn v2_reuses_source_validation() {
        let mut result = completed();
        result.task_id = "not-registered".into();
        assert_eq!(
            result.validate(),
            Err(HumanoidBenchAdapterErrorV1::UnknownTask)
        );
        assert_eq!(
            result.to_observation_v2(subject()),
            Err(HumanoidBenchAdapterV2Error::InvalidSourceEvidence)
        );
    }
}
