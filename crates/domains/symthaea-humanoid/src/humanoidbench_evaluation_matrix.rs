// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predeclared HumanoidBench evaluation matrices and coverage receipts.
//!
//! This is a Symthaea evidence protocol, not an upstream HumanoidBench scoring
//! protocol. It prevents cherry-picked episodes from being represented as a
//! broader evaluation and deliberately does not define a cross-task score.

use crate::capability_observatory::CapabilitySubjectIdentityV1;
use crate::humanoidbench_observatory::{
    HUMANOIDBENCH_ROBOT_REGISTRY_V1, HUMANOIDBENCH_TASK_REGISTRY_V1,
    HUMANOIDBENCH_UPSTREAM_COMMIT, HumanoidBenchControlModeV1,
    HumanoidBenchEpisodeResultV1, HumanoidBenchExecutionStatusV1,
    humanoidbench_task_registry_commitment_v1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const HUMANOIDBENCH_EVALUATION_MATRIX_SCHEMA_V1: &str =
    "symthaea.humanoid.humanoidbench-evaluation-matrix.v1";
const MATRIX_COMMITMENT_DOMAIN_V1: &[u8] = b"symthaea:humanoidbench-evaluation-matrix:v1\0";
const PLANNED_EPISODE_DOMAIN_V1: &[u8] = b"symthaea:humanoidbench-planned-episode:v1\0";
const COVERAGE_RECEIPT_DOMAIN_V1: &[u8] = b"symthaea:humanoidbench-coverage-receipt:v1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidBenchSeedStateV1 {
    Specified(u64),
    Unspecified,
}

impl HumanoidBenchSeedStateV1 {
    fn matches(self, seed: Option<u64>) -> bool {
        match (self, seed) {
            (Self::Specified(expected), Some(actual)) => expected == actual,
            (Self::Unspecified, None) => true,
            _ => false,
        }
    }

    fn commit_into(self, hasher: &mut blake3::Hasher) {
        match self {
            Self::Specified(seed) => {
                hasher.update(&[1]);
                hasher.update(&seed.to_le_bytes());
            }
            Self::Unspecified => {
                hasher.update(&[0]);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HumanoidBenchTaskFamilyV1 {
    BalanceRecovery,
    Locomotion,
    MobileManipulation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchEvaluationCaseV1 {
    pub task_id: String,
    pub robot_id: String,
    pub control_mode: HumanoidBenchControlModeV1,
    pub seed: HumanoidBenchSeedStateV1,
    /// Exact observation/sensor configuration identity. This is intentionally
    /// opaque here; the runner artifact remains the detailed source evidence.
    pub observation_profile_id: String,
    /// Exact environment/configuration identity expected from the adapter.
    pub environment_id: String,
}

impl HumanoidBenchEvaluationCaseV1 {
    pub fn validate(&self) -> Result<(), HumanoidBenchMatrixErrorV1> {
        if !HUMANOIDBENCH_TASK_REGISTRY_V1.contains(&self.task_id.as_str()) {
            return Err(HumanoidBenchMatrixErrorV1::UnknownTask);
        }
        if !HUMANOIDBENCH_ROBOT_REGISTRY_V1.contains(&self.robot_id.as_str()) {
            return Err(HumanoidBenchMatrixErrorV1::UnknownRobot);
        }
        if self.control_mode != expected_control_for_robot(&self.robot_id) {
            return Err(HumanoidBenchMatrixErrorV1::ControlModeMismatch);
        }
        validate_id(
            &self.observation_profile_id,
            HumanoidBenchMatrixErrorV1::InvalidObservationProfileId,
        )?;
        validate_id(
            &self.environment_id,
            HumanoidBenchMatrixErrorV1::InvalidEnvironmentId,
        )?;
        Ok(())
    }

    pub fn task_family(&self) -> HumanoidBenchTaskFamilyV1 {
        task_family(&self.task_id)
    }

    fn semantic_commitment_v1(&self) -> Result<String, HumanoidBenchMatrixErrorV1> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:humanoidbench-evaluation-case:v1\0");
        self.commit_into(&mut hasher);
        Ok(format!("humanoidbench-case:{}", hasher.finalize().to_hex()))
    }

    fn commit_into(&self, hasher: &mut blake3::Hasher) {
        hash_str(hasher, &self.task_id);
        hash_str(hasher, &self.robot_id);
        hasher.update(&[control_mode_tag(self.control_mode)]);
        self.seed.commit_into(hasher);
        hash_str(hasher, &self.observation_profile_id);
        hash_str(hasher, &self.environment_id);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchEvaluationMatrixV1 {
    pub matrix_id: String,
    pub upstream_commit: String,
    pub task_registry_commitment: String,
    pub subject: CapabilitySubjectIdentityV1,
    pub authority_profile_id: String,
    pub evidence_profile_id: String,
    pub runner_config_ref: String,
    /// Ordered predeclared evaluation cases. Order is evidence-significant.
    pub cases: Vec<HumanoidBenchEvaluationCaseV1>,
}

impl HumanoidBenchEvaluationMatrixV1 {
    pub fn validate(&self) -> Result<(), HumanoidBenchMatrixErrorV1> {
        validate_id(&self.matrix_id, HumanoidBenchMatrixErrorV1::InvalidMatrixId)?;
        if self.upstream_commit != HUMANOIDBENCH_UPSTREAM_COMMIT {
            return Err(HumanoidBenchMatrixErrorV1::WrongUpstreamCommit);
        }
        if self.task_registry_commitment != humanoidbench_task_registry_commitment_v1() {
            return Err(HumanoidBenchMatrixErrorV1::WrongTaskRegistryCommitment);
        }
        validate_subject(&self.subject)?;
        validate_id(
            &self.authority_profile_id,
            HumanoidBenchMatrixErrorV1::InvalidAuthorityProfileId,
        )?;
        validate_id(
            &self.evidence_profile_id,
            HumanoidBenchMatrixErrorV1::InvalidEvidenceProfileId,
        )?;
        validate_reference(
            &self.runner_config_ref,
            HumanoidBenchMatrixErrorV1::InvalidRunnerConfigRef,
        )?;
        if self.cases.is_empty() {
            return Err(HumanoidBenchMatrixErrorV1::EmptyMatrix);
        }
        if self.cases.len() > 16_384 {
            return Err(HumanoidBenchMatrixErrorV1::MatrixTooLarge);
        }

        let mut semantic_cases = BTreeSet::new();
        for case in &self.cases {
            case.validate()?;
            let commitment = case.semantic_commitment_v1()?;
            if !semantic_cases.insert(commitment) {
                return Err(HumanoidBenchMatrixErrorV1::DuplicateCase);
            }
        }
        Ok(())
    }

    pub fn matrix_commitment_v1(&self) -> Result<String, HumanoidBenchMatrixErrorV1> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(MATRIX_COMMITMENT_DOMAIN_V1);
        hash_str(&mut hasher, HUMANOIDBENCH_EVALUATION_MATRIX_SCHEMA_V1);
        hash_str(&mut hasher, &self.matrix_id);
        hash_str(&mut hasher, &self.upstream_commit);
        hash_str(&mut hasher, &self.task_registry_commitment);
        commit_subject(&mut hasher, &self.subject);
        hash_str(&mut hasher, &self.authority_profile_id);
        hash_str(&mut hasher, &self.evidence_profile_id);
        hash_str(&mut hasher, &self.runner_config_ref);
        hasher.update(&(self.cases.len() as u32).to_le_bytes());
        for case in &self.cases {
            case.commit_into(&mut hasher);
        }
        Ok(format!(
            "humanoidbench-evaluation-matrix:{}",
            hasher.finalize().to_hex()
        ))
    }

    pub fn planned_episode_id_v1(
        &self,
        case_index: usize,
    ) -> Result<String, HumanoidBenchMatrixErrorV1> {
        self.validate()?;
        let case = self
            .cases
            .get(case_index)
            .ok_or(HumanoidBenchMatrixErrorV1::CaseIndexOutOfBounds)?;
        let matrix_commitment = self.matrix_commitment_v1()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PLANNED_EPISODE_DOMAIN_V1);
        hash_str(&mut hasher, &matrix_commitment);
        hasher.update(&(case_index as u64).to_le_bytes());
        case.commit_into(&mut hasher);
        Ok(format!(
            "humanoidbench-planned-episode:{}",
            hasher.finalize().to_hex()
        ))
    }

    pub fn planned_cases_v1(
        &self,
    ) -> Result<Vec<HumanoidBenchPlannedCaseV1>, HumanoidBenchMatrixErrorV1> {
        self.validate()?;
        let mut planned = Vec::with_capacity(self.cases.len());
        for (case_index, case) in self.cases.iter().enumerate() {
            planned.push(HumanoidBenchPlannedCaseV1 {
                case_index,
                planned_episode_id: self.planned_episode_id_v1(case_index)?,
                case: case.clone(),
            });
        }
        Ok(planned)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchPlannedCaseV1 {
    pub case_index: usize,
    pub planned_episode_id: String,
    pub case: HumanoidBenchEvaluationCaseV1,
}

/// Evidence wrapper emitted beside an imported episode result.
///
/// The wrapper binds the result back to the matrix and runner configuration
/// that existed before execution. It is not a replacement for the external
/// runner artifact referenced by `episode.runner_artifact_ref`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanoidBenchMatrixCaseResultV1 {
    pub matrix_commitment: String,
    pub runner_config_ref: String,
    pub observation_profile_id: String,
    pub subject: CapabilitySubjectIdentityV1,
    pub episode: HumanoidBenchEpisodeResultV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidBenchMatrixCoverageStatusV1 {
    CompletePredeclaredMatrix,
    PartialPredeclaredMatrix,
    InvalidMatrixEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HumanoidBenchMatrixEvidenceViolationV1 {
    DuplicateEpisode,
    UnplannedEpisode,
    SourceEvidenceInvalid,
    MatrixCommitmentMismatch,
    RunnerConfigMismatch,
    SubjectMismatch,
    ObservationProfileMismatch,
    TaskMismatch,
    RobotMismatch,
    ControlModeMismatch,
    SeedMismatch,
    EnvironmentMismatch,
    AuthorityProfileMismatch,
    EvidenceProfileMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchMatrixViolationEventV1 {
    pub episode_id: String,
    pub violation: HumanoidBenchMatrixEvidenceViolationV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchFamilyCoverageV1 {
    pub family: HumanoidBenchTaskFamilyV1,
    pub expected_cases: u64,
    pub admitted_cases: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanoidBenchTaskReturnSampleV1 {
    pub planned_episode_id: String,
    pub episode_return: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanoidBenchPerTaskEvidenceV1 {
    pub task_id: String,
    pub admitted_cases: u64,
    pub completed_returns: Vec<HumanoidBenchTaskReturnSampleV1>,
    pub terminated_count: u64,
    pub truncated_count: u64,
    pub infrastructure_indeterminate_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanoidBenchMatrixCoverageReceiptV1 {
    pub matrix_commitment: String,
    pub status: HumanoidBenchMatrixCoverageStatusV1,
    pub expected_cases: u64,
    pub supplied_results: u64,
    pub admitted_cases: u64,
    pub missing_planned_episode_ids: Vec<String>,
    pub violations: Vec<HumanoidBenchMatrixViolationEventV1>,
    pub family_coverage: Vec<HumanoidBenchFamilyCoverageV1>,
    pub per_task_evidence: Vec<HumanoidBenchPerTaskEvidenceV1>,
    pub receipt_commitment: String,
}

pub fn assess_humanoidbench_matrix_results_v1(
    matrix: &HumanoidBenchEvaluationMatrixV1,
    results: &[HumanoidBenchMatrixCaseResultV1],
) -> Result<HumanoidBenchMatrixCoverageReceiptV1, HumanoidBenchMatrixErrorV1> {
    matrix.validate()?;
    let matrix_commitment = matrix.matrix_commitment_v1()?;
    let planned = matrix.planned_cases_v1()?;

    let planned_by_id: BTreeMap<&str, &HumanoidBenchPlannedCaseV1> = planned
        .iter()
        .map(|planned_case| (planned_case.planned_episode_id.as_str(), planned_case))
        .collect();

    let mut seen_episode_ids = BTreeSet::new();
    let mut admitted_episode_ids = BTreeSet::new();
    let mut violations = Vec::new();
    let mut admitted: Vec<(&HumanoidBenchPlannedCaseV1, &HumanoidBenchMatrixCaseResultV1)> =
        Vec::new();

    for result in results {
        let episode_id = result.episode.episode_id.clone();
        if !seen_episode_ids.insert(episode_id.clone()) {
            violations.push(HumanoidBenchMatrixViolationEventV1 {
                episode_id,
                violation: HumanoidBenchMatrixEvidenceViolationV1::DuplicateEpisode,
            });
            continue;
        }

        let Some(planned_case) = planned_by_id.get(result.episode.episode_id.as_str()).copied()
        else {
            violations.push(HumanoidBenchMatrixViolationEventV1 {
                episode_id,
                violation: HumanoidBenchMatrixEvidenceViolationV1::UnplannedEpisode,
            });
            continue;
        };

        let case_violations = validate_case_result(matrix, &matrix_commitment, planned_case, result);
        if case_violations.is_empty() {
            admitted_episode_ids.insert(result.episode.episode_id.clone());
            admitted.push((planned_case, result));
        } else {
            for violation in case_violations {
                violations.push(HumanoidBenchMatrixViolationEventV1 {
                    episode_id: result.episode.episode_id.clone(),
                    violation,
                });
            }
        }
    }

    violations.sort_by(|left, right| {
        left.episode_id
            .cmp(&right.episode_id)
            .then(left.violation.cmp(&right.violation))
    });

    let missing_planned_episode_ids: Vec<String> = planned
        .iter()
        .filter(|planned_case| !admitted_episode_ids.contains(&planned_case.planned_episode_id))
        .map(|planned_case| planned_case.planned_episode_id.clone())
        .collect();

    let family_coverage = family_coverage(&planned, &admitted);
    let per_task_evidence = per_task_evidence(&admitted);

    let status = if !violations.is_empty() {
        HumanoidBenchMatrixCoverageStatusV1::InvalidMatrixEvidence
    } else if admitted.len() == planned.len() {
        HumanoidBenchMatrixCoverageStatusV1::CompletePredeclaredMatrix
    } else {
        HumanoidBenchMatrixCoverageStatusV1::PartialPredeclaredMatrix
    };

    let mut receipt = HumanoidBenchMatrixCoverageReceiptV1 {
        matrix_commitment,
        status,
        expected_cases: planned.len() as u64,
        supplied_results: results.len() as u64,
        admitted_cases: admitted.len() as u64,
        missing_planned_episode_ids,
        violations,
        family_coverage,
        per_task_evidence,
        receipt_commitment: String::new(),
    };
    receipt.receipt_commitment = coverage_receipt_commitment_v1(&receipt);
    Ok(receipt)
}

fn validate_case_result(
    matrix: &HumanoidBenchEvaluationMatrixV1,
    matrix_commitment: &str,
    planned: &HumanoidBenchPlannedCaseV1,
    result: &HumanoidBenchMatrixCaseResultV1,
) -> Vec<HumanoidBenchMatrixEvidenceViolationV1> {
    let mut violations = Vec::new();
    if result.episode.validate().is_err() {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::SourceEvidenceInvalid);
        return violations;
    }
    if result.matrix_commitment != matrix_commitment {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::MatrixCommitmentMismatch);
    }
    if result.runner_config_ref != matrix.runner_config_ref {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::RunnerConfigMismatch);
    }
    if result.subject != matrix.subject {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::SubjectMismatch);
    }
    if result.observation_profile_id != planned.case.observation_profile_id {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::ObservationProfileMismatch);
    }
    if result.episode.task_id != planned.case.task_id {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::TaskMismatch);
    }
    if result.episode.robot_id != planned.case.robot_id {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::RobotMismatch);
    }
    if result.episode.control_mode != planned.case.control_mode {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::ControlModeMismatch);
    }
    if !planned.case.seed.matches(result.episode.random_seed) {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::SeedMismatch);
    }
    if result.episode.environment_id != planned.case.environment_id {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::EnvironmentMismatch);
    }
    if result.episode.authority_profile_id != matrix.authority_profile_id {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::AuthorityProfileMismatch);
    }
    if result.episode.evidence_profile_id != matrix.evidence_profile_id {
        violations.push(HumanoidBenchMatrixEvidenceViolationV1::EvidenceProfileMismatch);
    }
    violations
}

fn family_coverage(
    planned: &[HumanoidBenchPlannedCaseV1],
    admitted: &[(&HumanoidBenchPlannedCaseV1, &HumanoidBenchMatrixCaseResultV1)],
) -> Vec<HumanoidBenchFamilyCoverageV1> {
    let mut expected = BTreeMap::<HumanoidBenchTaskFamilyV1, u64>::new();
    let mut actual = BTreeMap::<HumanoidBenchTaskFamilyV1, u64>::new();
    for planned_case in planned {
        *expected.entry(planned_case.case.task_family()).or_default() += 1;
    }
    for (planned_case, _) in admitted {
        *actual.entry(planned_case.case.task_family()).or_default() += 1;
    }
    expected
        .into_iter()
        .map(|(family, expected_cases)| HumanoidBenchFamilyCoverageV1 {
            family,
            expected_cases,
            admitted_cases: actual.get(&family).copied().unwrap_or(0),
        })
        .collect()
}

fn per_task_evidence(
    admitted: &[(&HumanoidBenchPlannedCaseV1, &HumanoidBenchMatrixCaseResultV1)],
) -> Vec<HumanoidBenchPerTaskEvidenceV1> {
    #[derive(Default)]
    struct Accumulator {
        admitted_cases: u64,
        completed_returns: Vec<HumanoidBenchTaskReturnSampleV1>,
        terminated_count: u64,
        truncated_count: u64,
        infrastructure_indeterminate_count: u64,
    }

    let mut by_task = BTreeMap::<String, Accumulator>::new();
    for (planned, result) in admitted {
        let accumulator = by_task.entry(planned.case.task_id.clone()).or_default();
        accumulator.admitted_cases += 1;
        match result.episode.execution_status {
            HumanoidBenchExecutionStatusV1::Completed => {
                if let Some(episode_return) = result.episode.episode_return {
                    accumulator.completed_returns.push(HumanoidBenchTaskReturnSampleV1 {
                        planned_episode_id: planned.planned_episode_id.clone(),
                        episode_return,
                    });
                }
                if result.episode.terminated == Some(true) {
                    accumulator.terminated_count += 1;
                }
                if result.episode.truncated == Some(true) {
                    accumulator.truncated_count += 1;
                }
            }
            HumanoidBenchExecutionStatusV1::InfrastructureFailure => {
                accumulator.infrastructure_indeterminate_count += 1;
            }
        }
    }

    by_task
        .into_iter()
        .map(|(task_id, mut accumulator)| {
            accumulator.completed_returns.sort_by(|left, right| {
                left.planned_episode_id.cmp(&right.planned_episode_id)
            });
            HumanoidBenchPerTaskEvidenceV1 {
                task_id,
                admitted_cases: accumulator.admitted_cases,
                completed_returns: accumulator.completed_returns,
                terminated_count: accumulator.terminated_count,
                truncated_count: accumulator.truncated_count,
                infrastructure_indeterminate_count: accumulator.infrastructure_indeterminate_count,
            }
        })
        .collect()
}

fn coverage_receipt_commitment_v1(receipt: &HumanoidBenchMatrixCoverageReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(COVERAGE_RECEIPT_DOMAIN_V1);
    hash_str(&mut hasher, &receipt.matrix_commitment);
    hasher.update(&[coverage_status_tag(receipt.status)]);
    hasher.update(&receipt.expected_cases.to_le_bytes());
    hasher.update(&receipt.supplied_results.to_le_bytes());
    hasher.update(&receipt.admitted_cases.to_le_bytes());
    commit_strings(&mut hasher, &receipt.missing_planned_episode_ids);
    hasher.update(&(receipt.violations.len() as u32).to_le_bytes());
    for violation in &receipt.violations {
        hash_str(&mut hasher, &violation.episode_id);
        hasher.update(&[violation_tag(violation.violation)]);
    }
    hasher.update(&(receipt.family_coverage.len() as u32).to_le_bytes());
    for coverage in &receipt.family_coverage {
        hasher.update(&[family_tag(coverage.family)]);
        hasher.update(&coverage.expected_cases.to_le_bytes());
        hasher.update(&coverage.admitted_cases.to_le_bytes());
    }
    hasher.update(&(receipt.per_task_evidence.len() as u32).to_le_bytes());
    for task in &receipt.per_task_evidence {
        hash_str(&mut hasher, &task.task_id);
        hasher.update(&task.admitted_cases.to_le_bytes());
        hasher.update(&(task.completed_returns.len() as u32).to_le_bytes());
        for sample in &task.completed_returns {
            hash_str(&mut hasher, &sample.planned_episode_id);
            hasher.update(&sample.episode_return.to_bits().to_le_bytes());
        }
        hasher.update(&task.terminated_count.to_le_bytes());
        hasher.update(&task.truncated_count.to_le_bytes());
        hasher.update(&task.infrastructure_indeterminate_count.to_le_bytes());
    }
    format!(
        "humanoidbench-coverage-receipt:{}",
        hasher.finalize().to_hex()
    )
}

fn expected_control_for_robot(robot_id: &str) -> HumanoidBenchControlModeV1 {
    if robot_id == "g1" {
        HumanoidBenchControlModeV1::Torque
    } else {
        HumanoidBenchControlModeV1::Position
    }
}

fn task_family(task_id: &str) -> HumanoidBenchTaskFamilyV1 {
    match task_id {
        "balance_hard" | "balance_simple" => HumanoidBenchTaskFamilyV1::BalanceRecovery,
        "stand" | "walk" | "run" | "maze" | "hurdle" | "crawl" | "highbar_hard"
        | "highbar_simple" | "sit_hard" | "sit_simple" | "stair" | "slide" | "pole" => {
            HumanoidBenchTaskFamilyV1::Locomotion
        }
        _ => HumanoidBenchTaskFamilyV1::MobileManipulation,
    }
}

fn validate_subject(subject: &CapabilitySubjectIdentityV1) -> Result<(), HumanoidBenchMatrixErrorV1> {
    for value in [
        &subject.source_head,
        &subject.model_or_policy_id,
        &subject.morphology_id,
        &subject.sensor_actuator_profile_id,
    ] {
        validate_id(value, HumanoidBenchMatrixErrorV1::InvalidSubjectIdentity)?;
    }
    Ok(())
}

fn validate_id(value: &str, error: HumanoidBenchMatrixErrorV1) -> Result<(), HumanoidBenchMatrixErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 192 {
        Err(error)
    } else {
        Ok(())
    }
}

fn validate_reference(
    value: &str,
    error: HumanoidBenchMatrixErrorV1,
) -> Result<(), HumanoidBenchMatrixErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 512 {
        Err(error)
    } else {
        Ok(())
    }
}

fn commit_subject(hasher: &mut blake3::Hasher, subject: &CapabilitySubjectIdentityV1) {
    hash_str(hasher, &subject.source_head);
    hash_str(hasher, &subject.model_or_policy_id);
    hash_str(hasher, &subject.morphology_id);
    hash_str(hasher, &subject.sensor_actuator_profile_id);
}

fn commit_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    hasher.update(&(values.len() as u32).to_le_bytes());
    for value in values {
        hash_str(hasher, value);
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

fn control_mode_tag(value: HumanoidBenchControlModeV1) -> u8 {
    match value {
        HumanoidBenchControlModeV1::Position => 0,
        HumanoidBenchControlModeV1::Torque => 1,
    }
}

fn family_tag(value: HumanoidBenchTaskFamilyV1) -> u8 {
    match value {
        HumanoidBenchTaskFamilyV1::BalanceRecovery => 0,
        HumanoidBenchTaskFamilyV1::Locomotion => 1,
        HumanoidBenchTaskFamilyV1::MobileManipulation => 2,
    }
}

fn coverage_status_tag(value: HumanoidBenchMatrixCoverageStatusV1) -> u8 {
    match value {
        HumanoidBenchMatrixCoverageStatusV1::CompletePredeclaredMatrix => 0,
        HumanoidBenchMatrixCoverageStatusV1::PartialPredeclaredMatrix => 1,
        HumanoidBenchMatrixCoverageStatusV1::InvalidMatrixEvidence => 2,
    }
}

fn violation_tag(value: HumanoidBenchMatrixEvidenceViolationV1) -> u8 {
    value as u8
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanoidBenchMatrixErrorV1 {
    InvalidMatrixId,
    WrongUpstreamCommit,
    WrongTaskRegistryCommitment,
    InvalidSubjectIdentity,
    InvalidAuthorityProfileId,
    InvalidEvidenceProfileId,
    InvalidRunnerConfigRef,
    EmptyMatrix,
    MatrixTooLarge,
    DuplicateCase,
    UnknownTask,
    UnknownRobot,
    ControlModeMismatch,
    InvalidObservationProfileId,
    InvalidEnvironmentId,
    CaseIndexOutOfBounds,
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

    fn case(task_id: &str, seed: HumanoidBenchSeedStateV1) -> HumanoidBenchEvaluationCaseV1 {
        HumanoidBenchEvaluationCaseV1 {
            task_id: task_id.into(),
            robot_id: "h1hand".into(),
            control_mode: HumanoidBenchControlModeV1::Position,
            seed,
            observation_profile_id: "privileged-state-v1".into(),
            environment_id: "mujoco-upstream".into(),
        }
    }

    fn matrix() -> HumanoidBenchEvaluationMatrixV1 {
        HumanoidBenchEvaluationMatrixV1 {
            matrix_id: "matrix-v1".into(),
            upstream_commit: HUMANOIDBENCH_UPSTREAM_COMMIT.into(),
            task_registry_commitment: humanoidbench_task_registry_commitment_v1(),
            subject: subject(),
            authority_profile_id: "simulation-no-human-authority".into(),
            evidence_profile_id: "predeclared-matrix-v1".into(),
            runner_config_ref: "runner-config:exact-v1".into(),
            cases: vec![
                case("walk", HumanoidBenchSeedStateV1::Specified(0)),
                case("push", HumanoidBenchSeedStateV1::Specified(1)),
            ],
        }
    }

    fn result_for(
        matrix: &HumanoidBenchEvaluationMatrixV1,
        case_index: usize,
        episode_return: f64,
    ) -> HumanoidBenchMatrixCaseResultV1 {
        let planned = matrix.planned_cases_v1().unwrap();
        let planned_case = &planned[case_index];
        HumanoidBenchMatrixCaseResultV1 {
            matrix_commitment: matrix.matrix_commitment_v1().unwrap(),
            runner_config_ref: matrix.runner_config_ref.clone(),
            observation_profile_id: planned_case.case.observation_profile_id.clone(),
            subject: matrix.subject.clone(),
            episode: HumanoidBenchEpisodeResultV1 {
                upstream_commit: HUMANOIDBENCH_UPSTREAM_COMMIT.into(),
                task_id: planned_case.case.task_id.clone(),
                robot_id: planned_case.case.robot_id.clone(),
                control_mode: planned_case.case.control_mode,
                episode_id: planned_case.planned_episode_id.clone(),
                random_seed: match planned_case.case.seed {
                    HumanoidBenchSeedStateV1::Specified(seed) => Some(seed),
                    HumanoidBenchSeedStateV1::Unspecified => None,
                },
                execution_status: HumanoidBenchExecutionStatusV1::Completed,
                episode_return: Some(episode_return),
                episode_length_steps: Some(1000),
                terminated: Some(false),
                truncated: Some(true),
                infrastructure_failure: None,
                runner_artifact_ref: format!("runner-artifact:{case_index}"),
                environment_id: planned_case.case.environment_id.clone(),
                authority_profile_id: matrix.authority_profile_id.clone(),
                evidence_profile_id: matrix.evidence_profile_id.clone(),
            },
        }
    }

    #[test]
    fn matrix_commitment_is_deterministic_but_case_order_sensitive() {
        let a = matrix();
        let mut b = a.clone();
        assert_eq!(a.matrix_commitment_v1().unwrap(), b.matrix_commitment_v1().unwrap());
        b.cases.reverse();
        assert_ne!(a.matrix_commitment_v1().unwrap(), b.matrix_commitment_v1().unwrap());
    }

    #[test]
    fn duplicate_cases_fail_closed() {
        let mut value = matrix();
        value.cases.push(value.cases[0].clone());
        assert_eq!(value.validate(), Err(HumanoidBenchMatrixErrorV1::DuplicateCase));
    }

    #[test]
    fn pinned_robot_control_semantics_are_enforced() {
        let mut value = matrix();
        value.cases[0].robot_id = "g1".into();
        assert_eq!(value.validate(), Err(HumanoidBenchMatrixErrorV1::ControlModeMismatch));
        value.cases[0].control_mode = HumanoidBenchControlModeV1::Torque;
        assert!(value.validate().is_ok());
    }

    #[test]
    fn seed_zero_and_unspecified_are_distinct_cases() {
        let mut value = matrix();
        value.cases = vec![
            case("walk", HumanoidBenchSeedStateV1::Specified(0)),
            case("walk", HumanoidBenchSeedStateV1::Unspecified),
        ];
        assert!(value.validate().is_ok());
        assert_ne!(
            value.planned_episode_id_v1(0).unwrap(),
            value.planned_episode_id_v1(1).unwrap()
        );
    }

    #[test]
    fn complete_matrix_reports_per_task_samples_without_cross_task_score() {
        let value = matrix();
        let results = vec![result_for(&value, 0, -100.0), result_for(&value, 1, 10.0)];
        let receipt = assess_humanoidbench_matrix_results_v1(&value, &results).unwrap();
        assert_eq!(
            receipt.status,
            HumanoidBenchMatrixCoverageStatusV1::CompletePredeclaredMatrix
        );
        assert_eq!(receipt.admitted_cases, 2);
        assert!(receipt.missing_planned_episode_ids.is_empty());
        assert_eq!(receipt.per_task_evidence.len(), 2);
        assert!(receipt.per_task_evidence.iter().any(|task| {
            task.task_id == "walk"
                && task.completed_returns.len() == 1
                && task.completed_returns[0].episode_return == -100.0
        }));
    }

    #[test]
    fn partial_matrix_exposes_missing_planned_episode() {
        let value = matrix();
        let results = vec![result_for(&value, 0, 1.0)];
        let receipt = assess_humanoidbench_matrix_results_v1(&value, &results).unwrap();
        assert_eq!(
            receipt.status,
            HumanoidBenchMatrixCoverageStatusV1::PartialPredeclaredMatrix
        );
        assert_eq!(receipt.admitted_cases, 1);
        assert_eq!(receipt.missing_planned_episode_ids.len(), 1);
    }

    #[test]
    fn result_substitution_marks_matrix_evidence_invalid() {
        let value = matrix();
        let mut result = result_for(&value, 0, 1.0);
        result.runner_config_ref = "different-runner-config".into();
        let receipt = assess_humanoidbench_matrix_results_v1(&value, &[result]).unwrap();
        assert_eq!(
            receipt.status,
            HumanoidBenchMatrixCoverageStatusV1::InvalidMatrixEvidence
        );
        assert!(receipt.violations.iter().any(|event| {
            event.violation == HumanoidBenchMatrixEvidenceViolationV1::RunnerConfigMismatch
        }));
    }

    #[test]
    fn unplanned_episode_is_invalid_not_extra_coverage() {
        let value = matrix();
        let mut result = result_for(&value, 0, 1.0);
        result.episode.episode_id = "unplanned".into();
        let receipt = assess_humanoidbench_matrix_results_v1(&value, &[result]).unwrap();
        assert_eq!(
            receipt.status,
            HumanoidBenchMatrixCoverageStatusV1::InvalidMatrixEvidence
        );
        assert_eq!(receipt.admitted_cases, 0);
        assert_eq!(receipt.missing_planned_episode_ids.len(), 2);
    }

    #[test]
    fn infrastructure_failure_is_admitted_as_indeterminate_execution_evidence() {
        let value = matrix();
        let mut result = result_for(&value, 0, 1.0);
        result.episode.execution_status = HumanoidBenchExecutionStatusV1::InfrastructureFailure;
        result.episode.episode_return = None;
        result.episode.episode_length_steps = None;
        result.episode.terminated = None;
        result.episode.truncated = None;
        result.episode.infrastructure_failure = Some("runner failed before episode start".into());
        let receipt = assess_humanoidbench_matrix_results_v1(&value, &[result]).unwrap();
        assert_eq!(receipt.admitted_cases, 1);
        assert_eq!(
            receipt.status,
            HumanoidBenchMatrixCoverageStatusV1::PartialPredeclaredMatrix
        );
        let walk = receipt
            .per_task_evidence
            .iter()
            .find(|task| task.task_id == "walk")
            .unwrap();
        assert_eq!(walk.infrastructure_indeterminate_count, 1);
        assert!(walk.completed_returns.is_empty());
    }
}
