// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! High-assurance HumanoidBench matrix admission.
//!
//! HUM-OBS-003B establishes predeclared metadata/coverage semantics. HUM-OBS-003C
//! establishes subject and execution provenance for one planned episode. This
//! module composes both without silently changing either lower-level contract.

use crate::humanoidbench_evaluation_matrix::{
    HumanoidBenchEvaluationMatrixV1, HumanoidBenchMatrixCaseResultV1,
    HumanoidBenchMatrixCoverageStatusV1, assess_humanoidbench_matrix_results_v1,
};
use crate::humanoidbench_execution_receipt::{
    HumanoidBenchExecutionDispositionV1, HumanoidBenchExecutionReceiptV1,
    HumanoidBenchRunnerSubjectV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const HUMANOIDBENCH_HIGH_ASSURANCE_SCHEMA_V1: &str =
    "symthaea.humanoid.humanoidbench-high-assurance-admission.v1";
const HIGH_ASSURANCE_RECEIPT_DOMAIN_V1: &[u8] =
    b"symthaea:humanoidbench-high-assurance-receipt:v1\0";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanoidBenchHighAssuranceCaseEvidenceV1 {
    pub result: HumanoidBenchMatrixCaseResultV1,
    pub runner_subject: Option<HumanoidBenchRunnerSubjectV1>,
    pub execution_receipt: Option<HumanoidBenchExecutionReceiptV1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HumanoidBenchHighAssuranceViolationV1 {
    MetadataAdmissionFailed,
    MissingRunnerSubject,
    MissingExecutionReceipt,
    RunnerSubjectInvalid,
    ExecutionReceiptInvalid,
    DuplicateExecutionReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchHighAssuranceViolationEventV1 {
    pub planned_episode_id: String,
    pub violation: HumanoidBenchHighAssuranceViolationV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidBenchHighAssuranceStatusV1 {
    CompleteHighAssuranceMatrix,
    PartialHighAssuranceMatrix,
    InvalidHighAssuranceEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidBenchAdmittedExecutionReceiptV1 {
    pub planned_episode_id: String,
    pub execution_receipt_commitment: String,
    pub disposition: HumanoidBenchExecutionDispositionV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanoidBenchHighAssuranceReceiptV1 {
    pub schema: String,
    pub matrix_commitment: String,
    pub metadata_coverage_receipt_commitment: String,
    pub metadata_coverage_status: HumanoidBenchMatrixCoverageStatusV1,
    pub status: HumanoidBenchHighAssuranceStatusV1,
    pub planned_cases: u64,
    pub supplied_evidence_tuples: u64,
    /// Exact count admitted by the unchanged HUM-OBS-003B assessor.
    pub metadata_admitted_cases_003b: u64,
    /// Metadata-valid cases that are also free of episode-level 003B violation
    /// events and therefore eligible for stronger provenance evaluation.
    pub metadata_eligible_cases: u64,
    pub execution_provenance_valid_cases: u64,
    pub completed_provenance_cases: u64,
    pub infrastructure_indeterminate_provenance_cases: u64,
    pub missing_provenance_episode_ids: Vec<String>,
    pub violations: Vec<HumanoidBenchHighAssuranceViolationEventV1>,
    pub admitted_execution_receipts: Vec<HumanoidBenchAdmittedExecutionReceiptV1>,
    pub receipt_commitment: String,
}

pub fn assess_humanoidbench_high_assurance_v1(
    matrix: &HumanoidBenchEvaluationMatrixV1,
    evidence: &[HumanoidBenchHighAssuranceCaseEvidenceV1],
) -> Result<HumanoidBenchHighAssuranceReceiptV1, HumanoidBenchHighAssuranceErrorV1> {
    matrix
        .validate()
        .map_err(|_| HumanoidBenchHighAssuranceErrorV1::InvalidMatrix)?;
    if evidence.len() > 16_384 {
        return Err(HumanoidBenchHighAssuranceErrorV1::TooManyEvidenceTuples);
    }

    let planned = matrix
        .planned_cases_v1()
        .map_err(|_| HumanoidBenchHighAssuranceErrorV1::InvalidMatrix)?;
    let matrix_commitment = matrix
        .matrix_commitment_v1()
        .map_err(|_| HumanoidBenchHighAssuranceErrorV1::InvalidMatrix)?;
    let results: Vec<HumanoidBenchMatrixCaseResultV1> =
        evidence.iter().map(|item| item.result.clone()).collect();
    let metadata_receipt = assess_humanoidbench_matrix_results_v1(matrix, &results)
        .map_err(|_| HumanoidBenchHighAssuranceErrorV1::MetadataAssessmentFailed)?;

    let planned_ids: BTreeSet<String> = planned
        .iter()
        .map(|case| case.planned_episode_id.clone())
        .collect();
    let metadata_missing: BTreeSet<String> = metadata_receipt
        .missing_planned_episode_ids
        .iter()
        .cloned()
        .collect();
    let metadata_violation_ids: BTreeSet<String> = metadata_receipt
        .violations
        .iter()
        .map(|event| event.episode_id.clone())
        .collect();
    let metadata_eligible_ids: BTreeSet<String> = planned_ids
        .iter()
        .filter(|id| !metadata_missing.contains(*id) && !metadata_violation_ids.contains(*id))
        .cloned()
        .collect();

    let mut violations = Vec::new();
    let mut seen_receipt_commitments = BTreeSet::new();
    let mut admitted_by_episode = BTreeMap::<String, HumanoidBenchAdmittedExecutionReceiptV1>::new();

    for item in evidence {
        let episode_id = item.result.episode.episode_id.clone();
        if !metadata_eligible_ids.contains(&episode_id) {
            violations.push(HumanoidBenchHighAssuranceViolationEventV1 {
                planned_episode_id: episode_id,
                violation: HumanoidBenchHighAssuranceViolationV1::MetadataAdmissionFailed,
            });
            continue;
        }

        let Some(subject) = item.runner_subject.as_ref() else {
            violations.push(HumanoidBenchHighAssuranceViolationEventV1 {
                planned_episode_id: episode_id,
                violation: HumanoidBenchHighAssuranceViolationV1::MissingRunnerSubject,
            });
            continue;
        };
        if subject.validate_against_matrix(matrix).is_err() {
            violations.push(HumanoidBenchHighAssuranceViolationEventV1 {
                planned_episode_id: episode_id,
                violation: HumanoidBenchHighAssuranceViolationV1::RunnerSubjectInvalid,
            });
            continue;
        }

        let Some(receipt) = item.execution_receipt.as_ref() else {
            violations.push(HumanoidBenchHighAssuranceViolationEventV1 {
                planned_episode_id: episode_id,
                violation: HumanoidBenchHighAssuranceViolationV1::MissingExecutionReceipt,
            });
            continue;
        };
        if !seen_receipt_commitments.insert(receipt.receipt_commitment.clone()) {
            violations.push(HumanoidBenchHighAssuranceViolationEventV1 {
                planned_episode_id: episode_id,
                violation: HumanoidBenchHighAssuranceViolationV1::DuplicateExecutionReceipt,
            });
            continue;
        }
        if receipt
            .validate_against_result(subject, matrix, &item.result.episode)
            .is_err()
        {
            violations.push(HumanoidBenchHighAssuranceViolationEventV1 {
                planned_episode_id: episode_id,
                violation: HumanoidBenchHighAssuranceViolationV1::ExecutionReceiptInvalid,
            });
            continue;
        }

        admitted_by_episode.insert(
            episode_id.clone(),
            HumanoidBenchAdmittedExecutionReceiptV1 {
                planned_episode_id: episode_id,
                execution_receipt_commitment: receipt.receipt_commitment.clone(),
                disposition: receipt.disposition,
            },
        );
    }

    violations.sort_by(|left, right| {
        left.planned_episode_id
            .cmp(&right.planned_episode_id)
            .then(left.violation.cmp(&right.violation))
    });

    let mut admitted_execution_receipts = Vec::new();
    let mut missing_provenance_episode_ids = Vec::new();
    for planned_case in &planned {
        if let Some(admitted) = admitted_by_episode.remove(&planned_case.planned_episode_id) {
            admitted_execution_receipts.push(admitted);
        } else if metadata_eligible_ids.contains(&planned_case.planned_episode_id) {
            missing_provenance_episode_ids.push(planned_case.planned_episode_id.clone());
        }
    }

    let completed_provenance_cases = admitted_execution_receipts
        .iter()
        .filter(|entry| entry.disposition == HumanoidBenchExecutionDispositionV1::Completed)
        .count() as u64;
    let infrastructure_indeterminate_provenance_cases = admitted_execution_receipts
        .iter()
        .filter(|entry| {
            entry.disposition == HumanoidBenchExecutionDispositionV1::InfrastructureFailure
        })
        .count() as u64;

    let hard_invalid = metadata_receipt.status == HumanoidBenchMatrixCoverageStatusV1::InvalidMatrixEvidence
        || violations.iter().any(|event| {
            !matches!(
                event.violation,
                HumanoidBenchHighAssuranceViolationV1::MissingRunnerSubject
                    | HumanoidBenchHighAssuranceViolationV1::MissingExecutionReceipt
            )
        });
    let provenance_valid_cases = admitted_execution_receipts.len() as u64;
    let status = if hard_invalid {
        HumanoidBenchHighAssuranceStatusV1::InvalidHighAssuranceEvidence
    } else if provenance_valid_cases == planned.len() as u64
        && metadata_receipt.status
            == HumanoidBenchMatrixCoverageStatusV1::CompletePredeclaredMatrix
    {
        HumanoidBenchHighAssuranceStatusV1::CompleteHighAssuranceMatrix
    } else {
        HumanoidBenchHighAssuranceStatusV1::PartialHighAssuranceMatrix
    };

    let mut receipt = HumanoidBenchHighAssuranceReceiptV1 {
        schema: HUMANOIDBENCH_HIGH_ASSURANCE_SCHEMA_V1.into(),
        matrix_commitment,
        metadata_coverage_receipt_commitment: metadata_receipt.receipt_commitment.clone(),
        metadata_coverage_status: metadata_receipt.status,
        status,
        planned_cases: planned.len() as u64,
        supplied_evidence_tuples: evidence.len() as u64,
        metadata_admitted_cases_003b: metadata_receipt.admitted_cases,
        metadata_eligible_cases: metadata_eligible_ids.len() as u64,
        execution_provenance_valid_cases: provenance_valid_cases,
        completed_provenance_cases,
        infrastructure_indeterminate_provenance_cases,
        missing_provenance_episode_ids,
        violations,
        admitted_execution_receipts,
        receipt_commitment: String::new(),
    };
    receipt.receipt_commitment = high_assurance_receipt_commitment_v1(&receipt);
    Ok(receipt)
}

fn high_assurance_receipt_commitment_v1(receipt: &HumanoidBenchHighAssuranceReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(HIGH_ASSURANCE_RECEIPT_DOMAIN_V1);
    hash_str(&mut hasher, &receipt.schema);
    hash_str(&mut hasher, &receipt.matrix_commitment);
    hash_str(&mut hasher, &receipt.metadata_coverage_receipt_commitment);
    hasher.update(&[metadata_status_tag(receipt.metadata_coverage_status)]);
    hasher.update(&[high_assurance_status_tag(receipt.status)]);
    hasher.update(&receipt.planned_cases.to_le_bytes());
    hasher.update(&receipt.supplied_evidence_tuples.to_le_bytes());
    hasher.update(&receipt.metadata_admitted_cases_003b.to_le_bytes());
    hasher.update(&receipt.metadata_eligible_cases.to_le_bytes());
    hasher.update(&receipt.execution_provenance_valid_cases.to_le_bytes());
    hasher.update(&receipt.completed_provenance_cases.to_le_bytes());
    hasher.update(
        &receipt
            .infrastructure_indeterminate_provenance_cases
            .to_le_bytes(),
    );
    commit_strings(&mut hasher, &receipt.missing_provenance_episode_ids);
    hasher.update(&(receipt.violations.len() as u32).to_le_bytes());
    for event in &receipt.violations {
        hash_str(&mut hasher, &event.planned_episode_id);
        hasher.update(&[event.violation as u8]);
    }
    hasher.update(&(receipt.admitted_execution_receipts.len() as u32).to_le_bytes());
    for entry in &receipt.admitted_execution_receipts {
        hash_str(&mut hasher, &entry.planned_episode_id);
        hash_str(&mut hasher, &entry.execution_receipt_commitment);
        hasher.update(&[disposition_tag(entry.disposition)]);
    }
    format!(
        "humanoidbench-high-assurance-receipt:{}",
        hasher.finalize().to_hex()
    )
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

fn commit_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    hasher.update(&(values.len() as u32).to_le_bytes());
    for value in values {
        hash_str(hasher, value);
    }
}

fn metadata_status_tag(status: HumanoidBenchMatrixCoverageStatusV1) -> u8 {
    match status {
        HumanoidBenchMatrixCoverageStatusV1::CompletePredeclaredMatrix => 0,
        HumanoidBenchMatrixCoverageStatusV1::PartialPredeclaredMatrix => 1,
        HumanoidBenchMatrixCoverageStatusV1::InvalidMatrixEvidence => 2,
    }
}

fn high_assurance_status_tag(status: HumanoidBenchHighAssuranceStatusV1) -> u8 {
    match status {
        HumanoidBenchHighAssuranceStatusV1::CompleteHighAssuranceMatrix => 0,
        HumanoidBenchHighAssuranceStatusV1::PartialHighAssuranceMatrix => 1,
        HumanoidBenchHighAssuranceStatusV1::InvalidHighAssuranceEvidence => 2,
    }
}

fn disposition_tag(disposition: HumanoidBenchExecutionDispositionV1) -> u8 {
    match disposition {
        HumanoidBenchExecutionDispositionV1::Completed => 0,
        HumanoidBenchExecutionDispositionV1::InfrastructureFailure => 1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanoidBenchHighAssuranceErrorV1 {
    InvalidMatrix,
    TooManyEvidenceTuples,
    MetadataAssessmentFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability_observatory::CapabilitySubjectIdentityV1;
    use crate::humanoidbench_evaluation_matrix::{
        HumanoidBenchEvaluationCaseV1, HumanoidBenchSeedStateV1,
    };
    use crate::humanoidbench_execution_receipt::{
        ExecutionEnvironmentIdentityV1, HumanoidBenchRunnerPhaseV1,
        humanoidbench_adapter_input_commitment_v1,
    };
    use crate::humanoidbench_observatory::{
        HUMANOIDBENCH_UPSTREAM_COMMIT, HumanoidBenchControlModeV1,
        HumanoidBenchEpisodeResultV1, HumanoidBenchExecutionStatusV1,
        humanoidbench_task_registry_commitment_v1,
    };

    fn digest(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn matrix() -> HumanoidBenchEvaluationMatrixV1 {
        HumanoidBenchEvaluationMatrixV1 {
            matrix_id: "high-assurance-matrix".into(),
            upstream_commit: HUMANOIDBENCH_UPSTREAM_COMMIT.into(),
            task_registry_commitment: humanoidbench_task_registry_commitment_v1(),
            subject: CapabilitySubjectIdentityV1 {
                source_head: "source-head".into(),
                model_or_policy_id: "policy-v1".into(),
                morphology_id: "morph-v1".into(),
                sensor_actuator_profile_id: "sim-profile-v1".into(),
            },
            authority_profile_id: "simulation-no-human-authority".into(),
            evidence_profile_id: "high-assurance-v1".into(),
            runner_config_ref: "artifact:runner-config".into(),
            cases: vec![
                HumanoidBenchEvaluationCaseV1 {
                    task_id: "walk".into(),
                    robot_id: "h1hand".into(),
                    control_mode: HumanoidBenchControlModeV1::Position,
                    seed: HumanoidBenchSeedStateV1::Specified(7),
                    observation_profile_id: "obs-v1".into(),
                    environment_id: "env-v1".into(),
                },
                HumanoidBenchEvaluationCaseV1 {
                    task_id: "push".into(),
                    robot_id: "h1hand".into(),
                    control_mode: HumanoidBenchControlModeV1::Position,
                    seed: HumanoidBenchSeedStateV1::Specified(8),
                    observation_profile_id: "obs-v1".into(),
                    environment_id: "env-v1".into(),
                },
            ],
        }
    }

    fn runner_subject(
        matrix: &HumanoidBenchEvaluationMatrixV1,
        case_index: usize,
    ) -> HumanoidBenchRunnerSubjectV1 {
        let case = &matrix.cases[case_index];
        HumanoidBenchRunnerSubjectV1 {
            matrix_commitment: matrix.matrix_commitment_v1().unwrap(),
            case_index: case_index as u64,
            planned_episode_id: matrix.planned_episode_id_v1(case_index).unwrap(),
            source_head: matrix.subject.source_head.clone(),
            model_or_policy_id: matrix.subject.model_or_policy_id.clone(),
            model_or_policy_artifact_ref: "artifact:policy".into(),
            model_or_policy_commitment: digest("policy"),
            morphology_id: matrix.subject.morphology_id.clone(),
            morphology_artifact_ref: "artifact:morph".into(),
            morphology_commitment: digest("morph"),
            sensor_actuator_profile_id: matrix.subject.sensor_actuator_profile_id.clone(),
            simulation_profile_ref: "artifact:simulation-profile".into(),
            simulation_profile_commitment: digest("simulation-profile"),
            upstream_commit: matrix.upstream_commit.clone(),
            task_id: case.task_id.clone(),
            robot_id: case.robot_id.clone(),
            control_mode: case.control_mode,
            seed: case.seed,
            observation_profile_id: case.observation_profile_id.clone(),
            environment_id: case.environment_id.clone(),
            runner_artifact_ref: "artifact:runner".into(),
            runner_artifact_commitment: digest("runner"),
            runner_config_ref: matrix.runner_config_ref.clone(),
            runner_config_commitment: digest("runner-config"),
            mujoco_runtime_ref: "artifact:mujoco".into(),
            mujoco_runtime_commitment: digest("mujoco"),
            execution_environment: ExecutionEnvironmentIdentityV1::Hermetic {
                environment_ref: "nix:humanoidbench".into(),
                environment_commitment: digest("nix-env"),
            },
        }
    }

    fn result(
        matrix: &HumanoidBenchEvaluationMatrixV1,
        subject: &HumanoidBenchRunnerSubjectV1,
        status: HumanoidBenchExecutionStatusV1,
    ) -> HumanoidBenchMatrixCaseResultV1 {
        let completed = status == HumanoidBenchExecutionStatusV1::Completed;
        HumanoidBenchMatrixCaseResultV1 {
            matrix_commitment: matrix.matrix_commitment_v1().unwrap(),
            runner_config_ref: matrix.runner_config_ref.clone(),
            observation_profile_id: subject.observation_profile_id.clone(),
            subject: matrix.subject.clone(),
            episode: HumanoidBenchEpisodeResultV1 {
                upstream_commit: subject.upstream_commit.clone(),
                task_id: subject.task_id.clone(),
                robot_id: subject.robot_id.clone(),
                control_mode: subject.control_mode,
                episode_id: subject.planned_episode_id.clone(),
                random_seed: match subject.seed {
                    HumanoidBenchSeedStateV1::Specified(seed) => Some(seed),
                    HumanoidBenchSeedStateV1::Unspecified => None,
                },
                execution_status: status,
                episode_return: completed.then_some(1.0),
                episode_length_steps: completed.then_some(100),
                terminated: completed.then_some(false),
                truncated: completed.then_some(true),
                infrastructure_failure: (!completed).then_some("runner-lost".into()),
                runner_artifact_ref: subject.runner_artifact_ref.clone(),
                environment_id: subject.environment_id.clone(),
                authority_profile_id: matrix.authority_profile_id.clone(),
                evidence_profile_id: matrix.evidence_profile_id.clone(),
            },
        }
    }

    fn evidence(
        matrix: &HumanoidBenchEvaluationMatrixV1,
        case_index: usize,
        disposition: HumanoidBenchExecutionDispositionV1,
    ) -> HumanoidBenchHighAssuranceCaseEvidenceV1 {
        let subject = runner_subject(matrix, case_index);
        let status = match disposition {
            HumanoidBenchExecutionDispositionV1::Completed => {
                HumanoidBenchExecutionStatusV1::Completed
            }
            HumanoidBenchExecutionDispositionV1::InfrastructureFailure => {
                HumanoidBenchExecutionStatusV1::InfrastructureFailure
            }
        };
        let result = result(matrix, &subject, status);
        let adapter = humanoidbench_adapter_input_commitment_v1(&result.episode).unwrap();
        let (phase, raw_ref, raw_commitment) = match disposition {
            HumanoidBenchExecutionDispositionV1::Completed => (
                HumanoidBenchRunnerPhaseV1::EpisodeCompleted,
                Some(format!("artifact:raw-result-{case_index}")),
                Some(digest(&format!("raw-result-{case_index}"))),
            ),
            HumanoidBenchExecutionDispositionV1::InfrastructureFailure => (
                HumanoidBenchRunnerPhaseV1::EpisodeStarted,
                None,
                None,
            ),
        };
        let receipt = HumanoidBenchExecutionReceiptV1::new(
            &subject,
            matrix,
            format!("execution-{case_index}"),
            "monotonic-ns-v1",
            100,
            200,
            phase,
            disposition,
            raw_ref,
            raw_commitment,
            Some(adapter),
            vec![format!("artifact:log-{case_index}")],
        )
        .unwrap();
        HumanoidBenchHighAssuranceCaseEvidenceV1 {
            result,
            runner_subject: Some(subject),
            execution_receipt: Some(receipt),
        }
    }

    #[test]
    fn complete_valid_tuples_establish_complete_provenance_not_performance() {
        let matrix = matrix();
        let evidence = vec![
            evidence(&matrix, 0, HumanoidBenchExecutionDispositionV1::Completed),
            evidence(&matrix, 1, HumanoidBenchExecutionDispositionV1::Completed),
        ];
        let receipt = assess_humanoidbench_high_assurance_v1(&matrix, &evidence).unwrap();
        assert_eq!(
            receipt.status,
            HumanoidBenchHighAssuranceStatusV1::CompleteHighAssuranceMatrix
        );
        assert_eq!(receipt.execution_provenance_valid_cases, 2);
        assert_eq!(receipt.completed_provenance_cases, 2);
        assert!(!receipt.receipt_commitment.is_empty());
    }

    #[test]
    fn missing_receipt_is_partial_not_fabricated_failure() {
        let matrix = matrix();
        let mut first = evidence(&matrix, 0, HumanoidBenchExecutionDispositionV1::Completed);
        first.execution_receipt = None;
        let receipt = assess_humanoidbench_high_assurance_v1(&matrix, &[first]).unwrap();
        assert_eq!(
            receipt.status,
            HumanoidBenchHighAssuranceStatusV1::PartialHighAssuranceMatrix
        );
        assert_eq!(receipt.execution_provenance_valid_cases, 0);
        assert_eq!(receipt.missing_provenance_episode_ids.len(), 1);
        assert!(receipt.violations.iter().any(|event| {
            event.violation == HumanoidBenchHighAssuranceViolationV1::MissingExecutionReceipt
        }));
    }

    #[test]
    fn subject_substitution_is_invalid_high_assurance_evidence() {
        let matrix = matrix();
        let mut item = evidence(&matrix, 0, HumanoidBenchExecutionDispositionV1::Completed);
        item.runner_subject
            .as_mut()
            .unwrap()
            .model_or_policy_id = "other-policy".into();
        let receipt = assess_humanoidbench_high_assurance_v1(&matrix, &[item]).unwrap();
        assert_eq!(
            receipt.status,
            HumanoidBenchHighAssuranceStatusV1::InvalidHighAssuranceEvidence
        );
        assert_eq!(receipt.execution_provenance_valid_cases, 0);
    }

    #[test]
    fn valid_infrastructure_failure_proves_attempt_without_performance() {
        let matrix = matrix();
        let item = evidence(
            &matrix,
            0,
            HumanoidBenchExecutionDispositionV1::InfrastructureFailure,
        );
        let receipt = assess_humanoidbench_high_assurance_v1(&matrix, &[item]).unwrap();
        assert_eq!(receipt.execution_provenance_valid_cases, 1);
        assert_eq!(receipt.completed_provenance_cases, 0);
        assert_eq!(receipt.infrastructure_indeterminate_provenance_cases, 1);
    }

    #[test]
    fn duplicate_receipt_cannot_increase_provenance_coverage() {
        let matrix = matrix();
        let first = evidence(&matrix, 0, HumanoidBenchExecutionDispositionV1::Completed);
        let mut second = evidence(&matrix, 1, HumanoidBenchExecutionDispositionV1::Completed);
        second.execution_receipt = first.execution_receipt.clone();
        let receipt = assess_humanoidbench_high_assurance_v1(&matrix, &[first, second]).unwrap();
        assert_eq!(
            receipt.status,
            HumanoidBenchHighAssuranceStatusV1::InvalidHighAssuranceEvidence
        );
        assert_eq!(receipt.execution_provenance_valid_cases, 1);
    }

    #[test]
    fn receipt_identity_is_deterministic_for_same_evidence() {
        let matrix = matrix();
        let evidence = vec![evidence(
            &matrix,
            0,
            HumanoidBenchExecutionDispositionV1::Completed,
        )];
        let a = assess_humanoidbench_high_assurance_v1(&matrix, &evidence).unwrap();
        let b = assess_humanoidbench_high_assurance_v1(&matrix, &evidence).unwrap();
        assert_eq!(a.receipt_commitment, b.receipt_commitment);
    }
}
