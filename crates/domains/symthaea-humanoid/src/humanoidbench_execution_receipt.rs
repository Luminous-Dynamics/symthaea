// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hermetic-ish subject manifests and execution receipts for HumanoidBench.
//!
//! This layer closes the provenance gap between a predeclared matrix and an
//! imported benchmark result. It proves what the governed runner *claims* it
//! executed and binds that claim to exact artifacts; it does not prove physical
//! capability, sim-to-real transfer, or deployment authority.

use crate::humanoidbench_evaluation_matrix::{
    HumanoidBenchEvaluationMatrixV1, HumanoidBenchSeedStateV1,
};
use crate::humanoidbench_observatory::{
    HUMANOIDBENCH_UPSTREAM_COMMIT, HumanoidBenchControlModeV1,
    HumanoidBenchEpisodeResultV1, HumanoidBenchExecutionStatusV1,
};
use serde::{Deserialize, Serialize};

pub const HUMANOIDBENCH_EXECUTION_RECEIPT_SCHEMA_V1: &str =
    "symthaea.humanoid.humanoidbench-execution-receipt.v1";
const SUBJECT_DOMAIN_V1: &[u8] = b"symthaea:humanoidbench-runner-subject:v1\0";
const RECEIPT_DOMAIN_V1: &[u8] = b"symthaea:humanoidbench-execution-receipt:v1\0";
const ADAPTER_INPUT_DOMAIN_V1: &[u8] = b"symthaea:humanoidbench-adapter-input:v1\0";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionEnvironmentIdentityV1 {
    Hermetic {
        environment_ref: String,
        environment_commitment: String,
    },
    ResidualUncertainty {
        environment_ref: String,
        environment_commitment: String,
        uncertainty_ref: String,
    },
}

impl ExecutionEnvironmentIdentityV1 {
    fn validate(&self) -> Result<(), HumanoidBenchExecutionReceiptErrorV1> {
        match self {
            Self::Hermetic {
                environment_ref,
                environment_commitment,
            } => {
                validate_ref(environment_ref)?;
                validate_commitment(environment_commitment)
            }
            Self::ResidualUncertainty {
                environment_ref,
                environment_commitment,
                uncertainty_ref,
            } => {
                validate_ref(environment_ref)?;
                validate_commitment(environment_commitment)?;
                validate_ref(uncertainty_ref)
            }
        }
    }

    fn commit_into(&self, hasher: &mut blake3::Hasher) {
        match self {
            Self::Hermetic {
                environment_ref,
                environment_commitment,
            } => {
                hasher.update(&[0]);
                hash_str(hasher, environment_ref);
                hash_str(hasher, environment_commitment);
            }
            Self::ResidualUncertainty {
                environment_ref,
                environment_commitment,
                uncertainty_ref,
            } => {
                hasher.update(&[1]);
                hash_str(hasher, environment_ref);
                hash_str(hasher, environment_commitment);
                hash_str(hasher, uncertainty_ref);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchRunnerSubjectV1 {
    pub matrix_commitment: String,
    pub case_index: u64,
    pub planned_episode_id: String,

    pub source_head: String,
    pub model_or_policy_id: String,
    pub model_or_policy_artifact_ref: String,
    pub model_or_policy_commitment: String,
    pub morphology_id: String,
    pub morphology_artifact_ref: String,
    pub morphology_commitment: String,
    pub sensor_actuator_profile_id: String,
    pub simulation_profile_ref: String,
    pub simulation_profile_commitment: String,

    pub upstream_commit: String,
    pub task_id: String,
    pub robot_id: String,
    pub control_mode: HumanoidBenchControlModeV1,
    pub seed: HumanoidBenchSeedStateV1,
    pub observation_profile_id: String,
    pub environment_id: String,

    pub runner_artifact_ref: String,
    pub runner_artifact_commitment: String,
    pub runner_config_ref: String,
    pub runner_config_commitment: String,
    pub mujoco_runtime_ref: String,
    pub mujoco_runtime_commitment: String,
    pub execution_environment: ExecutionEnvironmentIdentityV1,
}

impl HumanoidBenchRunnerSubjectV1 {
    pub fn validate_against_matrix(
        &self,
        matrix: &HumanoidBenchEvaluationMatrixV1,
    ) -> Result<(), HumanoidBenchExecutionReceiptErrorV1> {
        matrix
            .validate()
            .map_err(|_| HumanoidBenchExecutionReceiptErrorV1::InvalidMatrix)?;
        let expected_matrix = matrix
            .matrix_commitment_v1()
            .map_err(|_| HumanoidBenchExecutionReceiptErrorV1::InvalidMatrix)?;
        if self.matrix_commitment != expected_matrix {
            return Err(HumanoidBenchExecutionReceiptErrorV1::MatrixCommitmentMismatch);
        }
        let case_index = usize::try_from(self.case_index)
            .map_err(|_| HumanoidBenchExecutionReceiptErrorV1::CaseIndexOutOfBounds)?;
        let case = matrix
            .cases
            .get(case_index)
            .ok_or(HumanoidBenchExecutionReceiptErrorV1::CaseIndexOutOfBounds)?;
        let expected_episode = matrix
            .planned_episode_id_v1(case_index)
            .map_err(|_| HumanoidBenchExecutionReceiptErrorV1::InvalidMatrix)?;
        if self.planned_episode_id != expected_episode {
            return Err(HumanoidBenchExecutionReceiptErrorV1::PlannedEpisodeMismatch);
        }

        if self.source_head != matrix.subject.source_head
            || self.model_or_policy_id != matrix.subject.model_or_policy_id
            || self.morphology_id != matrix.subject.morphology_id
            || self.sensor_actuator_profile_id != matrix.subject.sensor_actuator_profile_id
        {
            return Err(HumanoidBenchExecutionReceiptErrorV1::SubjectIdentityMismatch);
        }
        if self.runner_config_ref != matrix.runner_config_ref {
            return Err(HumanoidBenchExecutionReceiptErrorV1::RunnerConfigMismatch);
        }
        if self.upstream_commit != matrix.upstream_commit
            || self.upstream_commit != HUMANOIDBENCH_UPSTREAM_COMMIT
        {
            return Err(HumanoidBenchExecutionReceiptErrorV1::WrongUpstreamCommit);
        }
        if self.task_id != case.task_id
            || self.robot_id != case.robot_id
            || self.control_mode != case.control_mode
            || self.seed != case.seed
            || self.observation_profile_id != case.observation_profile_id
            || self.environment_id != case.environment_id
        {
            return Err(HumanoidBenchExecutionReceiptErrorV1::CaseSemanticsMismatch);
        }

        for value in [
            &self.source_head,
            &self.model_or_policy_id,
            &self.morphology_id,
            &self.sensor_actuator_profile_id,
            &self.task_id,
            &self.robot_id,
            &self.observation_profile_id,
            &self.environment_id,
        ] {
            validate_id(value)?;
        }
        for value in [
            &self.model_or_policy_artifact_ref,
            &self.morphology_artifact_ref,
            &self.simulation_profile_ref,
            &self.runner_artifact_ref,
            &self.runner_config_ref,
            &self.mujoco_runtime_ref,
        ] {
            validate_ref(value)?;
        }
        for value in [
            &self.model_or_policy_commitment,
            &self.morphology_commitment,
            &self.simulation_profile_commitment,
            &self.runner_artifact_commitment,
            &self.runner_config_commitment,
            &self.mujoco_runtime_commitment,
        ] {
            validate_commitment(value)?;
        }
        self.execution_environment.validate()?;
        Ok(())
    }

    pub fn subject_manifest_commitment_v1(
        &self,
        matrix: &HumanoidBenchEvaluationMatrixV1,
    ) -> Result<String, HumanoidBenchExecutionReceiptErrorV1> {
        self.validate_against_matrix(matrix)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(SUBJECT_DOMAIN_V1);
        hash_str(&mut hasher, HUMANOIDBENCH_EXECUTION_RECEIPT_SCHEMA_V1);
        hash_str(&mut hasher, &self.matrix_commitment);
        hasher.update(&self.case_index.to_le_bytes());
        hash_str(&mut hasher, &self.planned_episode_id);
        hash_str(&mut hasher, &self.source_head);
        hash_str(&mut hasher, &self.model_or_policy_id);
        hash_str(&mut hasher, &self.model_or_policy_artifact_ref);
        hash_str(&mut hasher, &self.model_or_policy_commitment);
        hash_str(&mut hasher, &self.morphology_id);
        hash_str(&mut hasher, &self.morphology_artifact_ref);
        hash_str(&mut hasher, &self.morphology_commitment);
        hash_str(&mut hasher, &self.sensor_actuator_profile_id);
        hash_str(&mut hasher, &self.simulation_profile_ref);
        hash_str(&mut hasher, &self.simulation_profile_commitment);
        hash_str(&mut hasher, &self.upstream_commit);
        hash_str(&mut hasher, &self.task_id);
        hash_str(&mut hasher, &self.robot_id);
        hasher.update(&[control_tag(self.control_mode)]);
        commit_seed(&mut hasher, self.seed);
        hash_str(&mut hasher, &self.observation_profile_id);
        hash_str(&mut hasher, &self.environment_id);
        hash_str(&mut hasher, &self.runner_artifact_ref);
        hash_str(&mut hasher, &self.runner_artifact_commitment);
        hash_str(&mut hasher, &self.runner_config_ref);
        hash_str(&mut hasher, &self.runner_config_commitment);
        hash_str(&mut hasher, &self.mujoco_runtime_ref);
        hash_str(&mut hasher, &self.mujoco_runtime_commitment);
        self.execution_environment.commit_into(&mut hasher);
        Ok(format!(
            "humanoidbench-runner-subject:{}",
            hasher.finalize().to_hex()
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidBenchRunnerPhaseV1 {
    Prepared,
    EnvironmentConstructed,
    EpisodeStarted,
    EpisodeCompleted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidBenchExecutionDispositionV1 {
    Completed,
    InfrastructureFailure,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchExecutionReceiptV1 {
    pub execution_id: String,
    pub subject_manifest_commitment: String,
    pub planned_episode_id: String,
    pub clock_domain_id: String,
    pub started_at_ns: u64,
    pub ended_at_ns: u64,
    pub last_established_phase: HumanoidBenchRunnerPhaseV1,
    pub disposition: HumanoidBenchExecutionDispositionV1,
    pub raw_result_artifact_ref: Option<String>,
    pub raw_result_artifact_commitment: Option<String>,
    pub adapter_input_commitment: Option<String>,
    pub log_refs: Vec<String>,
    pub receipt_commitment: String,
}

impl HumanoidBenchExecutionReceiptV1 {
    pub fn new(
        subject: &HumanoidBenchRunnerSubjectV1,
        matrix: &HumanoidBenchEvaluationMatrixV1,
        execution_id: impl Into<String>,
        clock_domain_id: impl Into<String>,
        started_at_ns: u64,
        ended_at_ns: u64,
        last_established_phase: HumanoidBenchRunnerPhaseV1,
        disposition: HumanoidBenchExecutionDispositionV1,
        raw_result_artifact_ref: Option<String>,
        raw_result_artifact_commitment: Option<String>,
        adapter_input_commitment: Option<String>,
        log_refs: Vec<String>,
    ) -> Result<Self, HumanoidBenchExecutionReceiptErrorV1> {
        let subject_manifest_commitment = subject.subject_manifest_commitment_v1(matrix)?;
        let mut receipt = Self {
            execution_id: execution_id.into(),
            subject_manifest_commitment,
            planned_episode_id: subject.planned_episode_id.clone(),
            clock_domain_id: clock_domain_id.into(),
            started_at_ns,
            ended_at_ns,
            last_established_phase,
            disposition,
            raw_result_artifact_ref,
            raw_result_artifact_commitment,
            adapter_input_commitment,
            log_refs,
            receipt_commitment: String::new(),
        };
        receipt.validate_shape()?;
        receipt.receipt_commitment = receipt.compute_commitment_v1();
        Ok(receipt)
    }

    pub fn validate_against_result(
        &self,
        subject: &HumanoidBenchRunnerSubjectV1,
        matrix: &HumanoidBenchEvaluationMatrixV1,
        result: &HumanoidBenchEpisodeResultV1,
    ) -> Result<(), HumanoidBenchExecutionReceiptErrorV1> {
        self.validate_shape()?;
        let expected_subject = subject.subject_manifest_commitment_v1(matrix)?;
        if self.subject_manifest_commitment != expected_subject
            || self.planned_episode_id != subject.planned_episode_id
        {
            return Err(HumanoidBenchExecutionReceiptErrorV1::SubjectManifestMismatch);
        }
        if self.receipt_commitment != self.compute_commitment_v1() {
            return Err(HumanoidBenchExecutionReceiptErrorV1::ReceiptCommitmentMismatch);
        }
        result
            .validate()
            .map_err(|_| HumanoidBenchExecutionReceiptErrorV1::InvalidAdapterInput)?;
        if result.episode_id != subject.planned_episode_id
            || result.upstream_commit != subject.upstream_commit
            || result.task_id != subject.task_id
            || result.robot_id != subject.robot_id
            || result.control_mode != subject.control_mode
            || result.random_seed != seed_option(subject.seed)
            || result.environment_id != subject.environment_id
            || result.runner_artifact_ref != subject.runner_artifact_ref
        {
            return Err(HumanoidBenchExecutionReceiptErrorV1::AdapterInputSubjectMismatch);
        }
        let expected_adapter = humanoidbench_adapter_input_commitment_v1(result)?;
        if self.adapter_input_commitment.as_deref() != Some(expected_adapter.as_str()) {
            return Err(HumanoidBenchExecutionReceiptErrorV1::AdapterInputCommitmentMismatch);
        }
        match (self.disposition, result.execution_status) {
            (
                HumanoidBenchExecutionDispositionV1::Completed,
                HumanoidBenchExecutionStatusV1::Completed,
            ) => Ok(()),
            (
                HumanoidBenchExecutionDispositionV1::InfrastructureFailure,
                HumanoidBenchExecutionStatusV1::InfrastructureFailure,
            ) => Ok(()),
            _ => Err(HumanoidBenchExecutionReceiptErrorV1::DispositionMismatch),
        }
    }

    fn validate_shape(&self) -> Result<(), HumanoidBenchExecutionReceiptErrorV1> {
        validate_id(&self.execution_id)?;
        validate_id(&self.planned_episode_id)?;
        validate_id(&self.clock_domain_id)?;
        validate_commitment(&self.subject_manifest_commitment)?;
        if self.ended_at_ns < self.started_at_ns {
            return Err(HumanoidBenchExecutionReceiptErrorV1::InvalidTimestamps);
        }
        if self.log_refs.len() > 64 {
            return Err(HumanoidBenchExecutionReceiptErrorV1::TooManyLogRefs);
        }
        for reference in &self.log_refs {
            validate_ref(reference)?;
        }
        match self.disposition {
            HumanoidBenchExecutionDispositionV1::Completed => {
                if self.last_established_phase != HumanoidBenchRunnerPhaseV1::EpisodeCompleted {
                    return Err(HumanoidBenchExecutionReceiptErrorV1::CompletedWithoutCompletionPhase);
                }
                require_result_fields(self)?;
            }
            HumanoidBenchExecutionDispositionV1::InfrastructureFailure => {
                if self.last_established_phase == HumanoidBenchRunnerPhaseV1::EpisodeCompleted {
                    return Err(HumanoidBenchExecutionReceiptErrorV1::FailureAfterCompletion);
                }
                if self.adapter_input_commitment.is_some() {
                    validate_commitment(
                        self.adapter_input_commitment
                            .as_deref()
                            .expect("checked Some"),
                    )?;
                }
                if let Some(reference) = &self.raw_result_artifact_ref {
                    validate_ref(reference)?;
                }
                if let Some(commitment) = &self.raw_result_artifact_commitment {
                    validate_commitment(commitment)?;
                }
            }
        }
        Ok(())
    }

    fn compute_commitment_v1(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(RECEIPT_DOMAIN_V1);
        hash_str(&mut hasher, HUMANOIDBENCH_EXECUTION_RECEIPT_SCHEMA_V1);
        hash_str(&mut hasher, &self.execution_id);
        hash_str(&mut hasher, &self.subject_manifest_commitment);
        hash_str(&mut hasher, &self.planned_episode_id);
        hash_str(&mut hasher, &self.clock_domain_id);
        hasher.update(&self.started_at_ns.to_le_bytes());
        hasher.update(&self.ended_at_ns.to_le_bytes());
        hasher.update(&[phase_tag(self.last_established_phase)]);
        hasher.update(&[disposition_tag(self.disposition)]);
        commit_optional_str(&mut hasher, self.raw_result_artifact_ref.as_deref());
        commit_optional_str(
            &mut hasher,
            self.raw_result_artifact_commitment.as_deref(),
        );
        commit_optional_str(&mut hasher, self.adapter_input_commitment.as_deref());
        let mut logs = self.log_refs.clone();
        logs.sort();
        hasher.update(&(logs.len() as u32).to_le_bytes());
        for value in logs {
            hash_str(&mut hasher, &value);
        }
        format!(
            "humanoidbench-execution-receipt:{}",
            hasher.finalize().to_hex()
        )
    }
}

pub fn humanoidbench_adapter_input_commitment_v1(
    result: &HumanoidBenchEpisodeResultV1,
) -> Result<String, HumanoidBenchExecutionReceiptErrorV1> {
    result
        .validate()
        .map_err(|_| HumanoidBenchExecutionReceiptErrorV1::InvalidAdapterInput)?;
    let bytes = serde_json::to_vec(result)
        .map_err(|_| HumanoidBenchExecutionReceiptErrorV1::AdapterInputSerializationFailed)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(ADAPTER_INPUT_DOMAIN_V1);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
    Ok(format!(
        "humanoidbench-adapter-input:{}",
        hasher.finalize().to_hex()
    ))
}

fn require_result_fields(
    receipt: &HumanoidBenchExecutionReceiptV1,
) -> Result<(), HumanoidBenchExecutionReceiptErrorV1> {
    let raw_ref = receipt
        .raw_result_artifact_ref
        .as_deref()
        .ok_or(HumanoidBenchExecutionReceiptErrorV1::CompletedMissingResultEvidence)?;
    let raw_commitment = receipt
        .raw_result_artifact_commitment
        .as_deref()
        .ok_or(HumanoidBenchExecutionReceiptErrorV1::CompletedMissingResultEvidence)?;
    let adapter = receipt
        .adapter_input_commitment
        .as_deref()
        .ok_or(HumanoidBenchExecutionReceiptErrorV1::CompletedMissingResultEvidence)?;
    validate_ref(raw_ref)?;
    validate_commitment(raw_commitment)?;
    validate_commitment(adapter)
}

fn seed_option(seed: HumanoidBenchSeedStateV1) -> Option<u64> {
    match seed {
        HumanoidBenchSeedStateV1::Specified(value) => Some(value),
        HumanoidBenchSeedStateV1::Unspecified => None,
    }
}

fn commit_seed(hasher: &mut blake3::Hasher, seed: HumanoidBenchSeedStateV1) {
    match seed {
        HumanoidBenchSeedStateV1::Specified(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        HumanoidBenchSeedStateV1::Unspecified => {
            hasher.update(&[0]);
        }
    }
}

fn control_tag(value: HumanoidBenchControlModeV1) -> u8 {
    match value {
        HumanoidBenchControlModeV1::Position => 0,
        HumanoidBenchControlModeV1::Torque => 1,
    }
}

fn phase_tag(value: HumanoidBenchRunnerPhaseV1) -> u8 {
    match value {
        HumanoidBenchRunnerPhaseV1::Prepared => 0,
        HumanoidBenchRunnerPhaseV1::EnvironmentConstructed => 1,
        HumanoidBenchRunnerPhaseV1::EpisodeStarted => 2,
        HumanoidBenchRunnerPhaseV1::EpisodeCompleted => 3,
    }
}

fn disposition_tag(value: HumanoidBenchExecutionDispositionV1) -> u8 {
    match value {
        HumanoidBenchExecutionDispositionV1::Completed => 0,
        HumanoidBenchExecutionDispositionV1::InfrastructureFailure => 1,
    }
}

fn validate_id(value: &str) -> Result<(), HumanoidBenchExecutionReceiptErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 256 {
        Err(HumanoidBenchExecutionReceiptErrorV1::InvalidIdentity)
    } else {
        Ok(())
    }
}

fn validate_ref(value: &str) -> Result<(), HumanoidBenchExecutionReceiptErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 512 {
        Err(HumanoidBenchExecutionReceiptErrorV1::InvalidReference)
    } else {
        Ok(())
    }
}

fn validate_commitment(value: &str) -> Result<(), HumanoidBenchExecutionReceiptErrorV1> {
    let Some((prefix, hex)) = value.rsplit_once(':') else {
        return Err(HumanoidBenchExecutionReceiptErrorV1::InvalidCommitment);
    };
    if prefix.trim().is_empty()
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(HumanoidBenchExecutionReceiptErrorV1::InvalidCommitment);
    }
    Ok(())
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

fn commit_optional_str(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_str(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanoidBenchExecutionReceiptErrorV1 {
    InvalidMatrix,
    MatrixCommitmentMismatch,
    CaseIndexOutOfBounds,
    PlannedEpisodeMismatch,
    SubjectIdentityMismatch,
    RunnerConfigMismatch,
    WrongUpstreamCommit,
    CaseSemanticsMismatch,
    InvalidIdentity,
    InvalidReference,
    InvalidCommitment,
    InvalidTimestamps,
    TooManyLogRefs,
    CompletedWithoutCompletionPhase,
    FailureAfterCompletion,
    CompletedMissingResultEvidence,
    SubjectManifestMismatch,
    ReceiptCommitmentMismatch,
    InvalidAdapterInput,
    AdapterInputSerializationFailed,
    AdapterInputSubjectMismatch,
    AdapterInputCommitmentMismatch,
    DispositionMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability_observatory::CapabilitySubjectIdentityV1;
    use crate::humanoidbench_evaluation_matrix::{
        HumanoidBenchEvaluationCaseV1, HumanoidBenchEvaluationMatrixV1,
    };

    fn digest(label: &str) -> String {
        format!("{label}:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn matrix() -> HumanoidBenchEvaluationMatrixV1 {
        let subject = CapabilitySubjectIdentityV1 {
            source_head: "source-head".into(),
            model_or_policy_id: "policy-v1".into(),
            morphology_id: "morph-v1".into(),
            sensor_actuator_profile_id: "sim-profile-v1".into(),
        };
        HumanoidBenchEvaluationMatrixV1 {
            matrix_id: "matrix-1".into(),
            upstream_commit: HUMANOIDBENCH_UPSTREAM_COMMIT.into(),
            task_registry_commitment:
                crate::humanoidbench_observatory::humanoidbench_task_registry_commitment_v1(),
            subject,
            authority_profile_id: "sim-no-human-authority".into(),
            evidence_profile_id: "external-import-only".into(),
            runner_config_ref: "artifact:runner-config".into(),
            cases: vec![HumanoidBenchEvaluationCaseV1 {
                task_id: "walk".into(),
                robot_id: "h1hand".into(),
                control_mode: HumanoidBenchControlModeV1::Position,
                seed: HumanoidBenchSeedStateV1::Specified(7),
                observation_profile_id: "obs-v1".into(),
                environment_id: "mujoco-env-v1".into(),
            }],
        }
    }

    fn subject(matrix: &HumanoidBenchEvaluationMatrixV1) -> HumanoidBenchRunnerSubjectV1 {
        HumanoidBenchRunnerSubjectV1 {
            matrix_commitment: matrix.matrix_commitment_v1().unwrap(),
            case_index: 0,
            planned_episode_id: matrix.planned_episode_id_v1(0).unwrap(),
            source_head: "source-head".into(),
            model_or_policy_id: "policy-v1".into(),
            model_or_policy_artifact_ref: "artifact:policy".into(),
            model_or_policy_commitment: digest("policy"),
            morphology_id: "morph-v1".into(),
            morphology_artifact_ref: "artifact:morph".into(),
            morphology_commitment: digest("morph"),
            sensor_actuator_profile_id: "sim-profile-v1".into(),
            simulation_profile_ref: "artifact:sim-profile".into(),
            simulation_profile_commitment: digest("sim-profile"),
            upstream_commit: HUMANOIDBENCH_UPSTREAM_COMMIT.into(),
            task_id: "walk".into(),
            robot_id: "h1hand".into(),
            control_mode: HumanoidBenchControlModeV1::Position,
            seed: HumanoidBenchSeedStateV1::Specified(7),
            observation_profile_id: "obs-v1".into(),
            environment_id: "mujoco-env-v1".into(),
            runner_artifact_ref: "artifact:runner".into(),
            runner_artifact_commitment: digest("runner"),
            runner_config_ref: "artifact:runner-config".into(),
            runner_config_commitment: digest("runner-config"),
            mujoco_runtime_ref: "artifact:mujoco".into(),
            mujoco_runtime_commitment: digest("mujoco"),
            execution_environment: ExecutionEnvironmentIdentityV1::Hermetic {
                environment_ref: "nix:runner-env".into(),
                environment_commitment: digest("runner-env"),
            },
        }
    }

    fn result(subject: &HumanoidBenchRunnerSubjectV1) -> HumanoidBenchEpisodeResultV1 {
        HumanoidBenchEpisodeResultV1 {
            upstream_commit: subject.upstream_commit.clone(),
            task_id: subject.task_id.clone(),
            robot_id: subject.robot_id.clone(),
            control_mode: subject.control_mode,
            episode_id: subject.planned_episode_id.clone(),
            random_seed: Some(7),
            execution_status: HumanoidBenchExecutionStatusV1::Completed,
            episode_return: Some(1.25),
            episode_length_steps: Some(1000),
            terminated: Some(false),
            truncated: Some(true),
            infrastructure_failure: None,
            runner_artifact_ref: subject.runner_artifact_ref.clone(),
            environment_id: subject.environment_id.clone(),
            authority_profile_id: "sim-no-human-authority".into(),
            evidence_profile_id: "external-import-only".into(),
        }
    }

    #[test]
    fn deterministic_subject_manifest_binds_exact_subject() {
        let matrix = matrix();
        let subject = subject(&matrix);
        let a = subject.subject_manifest_commitment_v1(&matrix).unwrap();
        let b = subject.subject_manifest_commitment_v1(&matrix).unwrap();
        assert_eq!(a, b);
        let mut changed = subject.clone();
        changed.model_or_policy_commitment = digest("different-policy");
        assert_ne!(
            a,
            changed.subject_manifest_commitment_v1(&matrix).unwrap()
        );
    }

    #[test]
    fn policy_identity_substitution_fails_closed() {
        let matrix = matrix();
        let mut subject = subject(&matrix);
        subject.model_or_policy_id = "other-policy".into();
        assert_eq!(
            subject.validate_against_matrix(&matrix),
            Err(HumanoidBenchExecutionReceiptErrorV1::SubjectIdentityMismatch)
        );
    }

    #[test]
    fn planned_episode_and_seed_are_exact() {
        let matrix = matrix();
        let mut subject = subject(&matrix);
        subject.seed = HumanoidBenchSeedStateV1::Specified(8);
        assert_eq!(
            subject.validate_against_matrix(&matrix),
            Err(HumanoidBenchExecutionReceiptErrorV1::CaseSemanticsMismatch)
        );
        let mut subject = subject(&matrix);
        subject.planned_episode_id = "other-episode".into();
        assert_eq!(
            subject.validate_against_matrix(&matrix),
            Err(HumanoidBenchExecutionReceiptErrorV1::PlannedEpisodeMismatch)
        );
    }

    #[test]
    fn completed_receipt_binds_adapter_input() {
        let matrix = matrix();
        let subject = subject(&matrix);
        let result = result(&subject);
        let adapter = humanoidbench_adapter_input_commitment_v1(&result).unwrap();
        let receipt = HumanoidBenchExecutionReceiptV1::new(
            &subject,
            &matrix,
            "execution-1",
            "monotonic-ns-v1",
            100,
            200,
            HumanoidBenchRunnerPhaseV1::EpisodeCompleted,
            HumanoidBenchExecutionDispositionV1::Completed,
            Some("artifact:raw-result".into()),
            Some(digest("raw-result")),
            Some(adapter),
            vec!["artifact:stdout".into()],
        )
        .unwrap();
        assert!(receipt
            .validate_against_result(&subject, &matrix, &result)
            .is_ok());
    }

    #[test]
    fn result_artifact_substitution_changes_receipt_commitment() {
        let matrix = matrix();
        let subject = subject(&matrix);
        let result = result(&subject);
        let adapter = humanoidbench_adapter_input_commitment_v1(&result).unwrap();
        let a = HumanoidBenchExecutionReceiptV1::new(
            &subject,
            &matrix,
            "execution-1",
            "clock",
            1,
            2,
            HumanoidBenchRunnerPhaseV1::EpisodeCompleted,
            HumanoidBenchExecutionDispositionV1::Completed,
            Some("artifact:raw-a".into()),
            Some(digest("raw-a")),
            Some(adapter.clone()),
            vec![],
        )
        .unwrap();
        let b = HumanoidBenchExecutionReceiptV1::new(
            &subject,
            &matrix,
            "execution-1",
            "clock",
            1,
            2,
            HumanoidBenchRunnerPhaseV1::EpisodeCompleted,
            HumanoidBenchExecutionDispositionV1::Completed,
            Some("artifact:raw-b".into()),
            Some(digest("raw-b")),
            Some(adapter),
            vec![],
        )
        .unwrap();
        assert_ne!(a.receipt_commitment, b.receipt_commitment);
    }

    #[test]
    fn infrastructure_failure_can_stop_before_episode_start() {
        let matrix = matrix();
        let subject = subject(&matrix);
        let receipt = HumanoidBenchExecutionReceiptV1::new(
            &subject,
            &matrix,
            "execution-failed",
            "clock",
            10,
            20,
            HumanoidBenchRunnerPhaseV1::Prepared,
            HumanoidBenchExecutionDispositionV1::InfrastructureFailure,
            None,
            None,
            None,
            vec!["artifact:stderr".into()],
        )
        .unwrap();
        assert_eq!(
            receipt.last_established_phase,
            HumanoidBenchRunnerPhaseV1::Prepared
        );
    }

    #[test]
    fn non_hermetic_environment_must_disclose_residual_uncertainty() {
        let matrix = matrix();
        let mut subject = subject(&matrix);
        subject.execution_environment = ExecutionEnvironmentIdentityV1::ResidualUncertainty {
            environment_ref: "python-env:lock".into(),
            environment_commitment: digest("python-env"),
            uncertainty_ref: "evidence:uncontrolled-gpu-driver".into(),
        };
        assert!(subject.validate_against_matrix(&matrix).is_ok());
        subject.execution_environment = ExecutionEnvironmentIdentityV1::ResidualUncertainty {
            environment_ref: "python-env:lock".into(),
            environment_commitment: digest("python-env"),
            uncertainty_ref: "".into(),
        };
        assert_eq!(
            subject.validate_against_matrix(&matrix),
            Err(HumanoidBenchExecutionReceiptErrorV1::InvalidReference)
        );
    }
}
