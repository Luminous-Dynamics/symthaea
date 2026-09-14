// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end adapter from the root GWT-1 experiment to psych-bench evidence.
//!
//! The root benchmark owns experiment execution and raw observations. This
//! module independently derives the typed receipt from those observations,
//! binds the canonical raw JSON bytes through the integrity envelope, and
//! returns the fail-closed qualification result. It does not promote a Butlin
//! support tier.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea::benchmarks::gwt1_specialist_qualification::{
    GWT1_RAW_OBSERVATION_SCHEMA_V1 as ROOT_GWT1_RAW_OBSERVATION_SCHEMA_V1,
    GWT1_SPECIALIZATION_CONTRACT_V1 as ROOT_GWT1_SPECIALIZATION_CONTRACT_V1,
    GWT1_TRAJECTORY_SCHEDULE_V1 as ROOT_GWT1_TRAJECTORY_SCHEDULE_V1,
    GWT1_TRAJECTORY_STEPS_V1 as ROOT_GWT1_TRAJECTORY_STEPS_V1, Gwt1PerturbationRawV1,
    Gwt1RawObservationsV1, Gwt1RunnerErrorV1, Gwt1SpecialistOutputMapV1,
    run_gwt1_specialist_qualification_v1,
};

use super::gwt1_evidence_envelope::{
    GWT1_EVIDENCE_ENVELOPE_SCHEMA_V1, GWT1_RAW_OBSERVATION_SCHEMA_V1,
    Gwt1EvidenceEnvelopeResolutionV1, Gwt1EvidenceEnvelopeV1, describe_raw_observations_v1,
    resolve_gwt1_evidence_envelope_v1,
};
use super::gwt1_qualification::{
    GWT1_QUALIFICATION_SCHEMA_V1, GWT1_SPECIALISTS_V1, Gwt1PerturbationObservationV1,
    Gwt1SpecialistIdentityV1, Gwt1SpecialistQualificationReceiptV1,
};

const EXPECTED_SPECIALIZATION_CONTRACT_V1: &str = "gwt1-specialization-matrix-v1";
const EXPECTED_TRAJECTORY_SCHEDULE_V1: &str = "gwt1-specialist-trajectory-v1-48";
const EXPECTED_TRAJECTORY_STEPS_V1: u32 = 48;

const EXPECTED_PERTURBATIONS_V1: [(&str, &str, &str, f64, f64); 4] = [
    (
        "drive-valence",
        "valence",
        "drive_manager",
        0.0_f32 as f64,
        0.80_f32 as f64,
    ),
    (
        "memory-unified-psi",
        "unified_psi",
        "memory_manager",
        0.40,
        0.80,
    ),
    (
        "learning-dissipative-health",
        "dissipative_health",
        "learning_manager",
        0.80,
        0.10,
    ),
    (
        "perception-phenomenal-binding",
        "phenomenal_binding",
        "perception_manager",
        0.50,
        0.90,
    ),
];

const IMPLEMENTATION_PATHS_V1: [(&str, &str); 4] = [
    (
        "drive_manager",
        "src/cognitive_loop/managers/drive_manager.rs",
    ),
    (
        "memory_manager",
        "src/cognitive_loop/managers/memory_manager.rs",
    ),
    (
        "learning_manager",
        "src/cognitive_loop/managers/learning_manager.rs",
    ),
    (
        "perception_manager",
        "src/cognitive_loop/managers/perception_manager.rs",
    ),
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1ExecutionIdentityV1 {
    pub source_commit_sha: String,
    pub source_tree_sha: String,
    pub execution_run_id: String,
    pub toolchain: String,
    pub specialist_blob_shas: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gwt1EndToEndEvidenceV1 {
    pub raw_observation_bytes: Vec<u8>,
    pub envelope: Gwt1EvidenceEnvelopeV1,
    pub resolution: Gwt1EvidenceEnvelopeResolutionV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gwt1EndToEndErrorV1 {
    Runner(String),
    RootContractDrift {
        field: String,
        observed: String,
        expected: String,
    },
    RawContractMismatch {
        field: String,
        observed: String,
        expected: String,
    },
    RawPerturbationContractMismatch {
        perturbation_id: String,
        field: String,
    },
    RawTrajectoryStepMismatch {
        index: usize,
        observed: u32,
        expected: u32,
    },
    RawOutputCoverageMismatch {
        context: String,
        observed: Vec<String>,
    },
    RawSerialization(String),
}

impl From<Gwt1RunnerErrorV1> for Gwt1EndToEndErrorV1 {
    fn from(error: Gwt1RunnerErrorV1) -> Self {
        Self::Runner(format!("{error:?}"))
    }
}

fn canonical_specialists() -> BTreeSet<String> {
    GWT1_SPECIALISTS_V1
        .iter()
        .map(|id| (*id).to_string())
        .collect()
}

fn validate_root_contract_v1() -> Result<(), Gwt1EndToEndErrorV1> {
    for (field, observed, expected) in [
        (
            "raw_observation_schema",
            ROOT_GWT1_RAW_OBSERVATION_SCHEMA_V1,
            GWT1_RAW_OBSERVATION_SCHEMA_V1,
        ),
        (
            "specialization_contract",
            ROOT_GWT1_SPECIALIZATION_CONTRACT_V1,
            EXPECTED_SPECIALIZATION_CONTRACT_V1,
        ),
        (
            "trajectory_schedule",
            ROOT_GWT1_TRAJECTORY_SCHEDULE_V1,
            EXPECTED_TRAJECTORY_SCHEDULE_V1,
        ),
    ] {
        if observed != expected {
            return Err(Gwt1EndToEndErrorV1::RootContractDrift {
                field: field.to_string(),
                observed: observed.to_string(),
                expected: expected.to_string(),
            });
        }
    }

    if ROOT_GWT1_TRAJECTORY_STEPS_V1 != EXPECTED_TRAJECTORY_STEPS_V1 {
        return Err(Gwt1EndToEndErrorV1::RootContractDrift {
            field: "trajectory_steps".to_string(),
            observed: ROOT_GWT1_TRAJECTORY_STEPS_V1.to_string(),
            expected: EXPECTED_TRAJECTORY_STEPS_V1.to_string(),
        });
    }

    Ok(())
}

fn validate_output_map(
    context: impl Into<String>,
    outputs: &Gwt1SpecialistOutputMapV1,
) -> Result<(), Gwt1EndToEndErrorV1> {
    let observed: BTreeSet<String> = outputs.keys().cloned().collect();
    if observed != canonical_specialists() {
        return Err(Gwt1EndToEndErrorV1::RawOutputCoverageMismatch {
            context: context.into(),
            observed: observed.into_iter().collect(),
        });
    }
    Ok(())
}

fn validate_raw_contract_v1(raw: &Gwt1RawObservationsV1) -> Result<(), Gwt1EndToEndErrorV1> {
    validate_root_contract_v1()?;

    for (field, observed, expected) in [
        (
            "schema",
            raw.schema.as_str(),
            GWT1_RAW_OBSERVATION_SCHEMA_V1,
        ),
        (
            "specialization_contract",
            raw.specialization_contract.as_str(),
            EXPECTED_SPECIALIZATION_CONTRACT_V1,
        ),
        (
            "trajectory_schedule",
            raw.trajectory_schedule.as_str(),
            EXPECTED_TRAJECTORY_SCHEDULE_V1,
        ),
    ] {
        if observed != expected {
            return Err(Gwt1EndToEndErrorV1::RawContractMismatch {
                field: field.to_string(),
                observed: observed.to_string(),
                expected: expected.to_string(),
            });
        }
    }

    if raw.trajectory.len() != EXPECTED_TRAJECTORY_STEPS_V1 as usize {
        return Err(Gwt1EndToEndErrorV1::RawContractMismatch {
            field: "trajectory_len".to_string(),
            observed: raw.trajectory.len().to_string(),
            expected: EXPECTED_TRAJECTORY_STEPS_V1.to_string(),
        });
    }

    for (index, step) in raw.trajectory.iter().enumerate() {
        let expected = index as u32;
        if step.step != expected {
            return Err(Gwt1EndToEndErrorV1::RawTrajectoryStepMismatch {
                index,
                observed: step.step,
                expected,
            });
        }
    }

    let observed_ids: BTreeSet<&str> = raw.perturbations.iter().map(|p| p.id.as_str()).collect();
    let expected_ids: BTreeSet<&str> = EXPECTED_PERTURBATIONS_V1
        .iter()
        .map(|(id, _, _, _, _)| *id)
        .collect();
    if raw.perturbations.len() != EXPECTED_PERTURBATIONS_V1.len() || observed_ids != expected_ids {
        return Err(Gwt1EndToEndErrorV1::RawContractMismatch {
            field: "perturbation_set".to_string(),
            observed: format!("{:?}", observed_ids),
            expected: format!("{:?}", expected_ids),
        });
    }

    for (id, field, target, baseline, perturbed) in EXPECTED_PERTURBATIONS_V1 {
        let observation = raw
            .perturbations
            .iter()
            .find(|observation| observation.id == id)
            .expect("validated perturbation set");

        if observation.field != field {
            return Err(Gwt1EndToEndErrorV1::RawPerturbationContractMismatch {
                perturbation_id: id.to_string(),
                field: "field".to_string(),
            });
        }
        if observation.target_specialist != target {
            return Err(Gwt1EndToEndErrorV1::RawPerturbationContractMismatch {
                perturbation_id: id.to_string(),
                field: "target_specialist".to_string(),
            });
        }
        if observation.baseline_value.to_bits() != baseline.to_bits() {
            return Err(Gwt1EndToEndErrorV1::RawPerturbationContractMismatch {
                perturbation_id: id.to_string(),
                field: "baseline_value".to_string(),
            });
        }
        if observation.perturbed_value.to_bits() != perturbed.to_bits() {
            return Err(Gwt1EndToEndErrorV1::RawPerturbationContractMismatch {
                perturbation_id: id.to_string(),
                field: "perturbed_value".to_string(),
            });
        }
    }

    Ok(())
}

fn changed_specialists(
    perturbation: &Gwt1PerturbationRawV1,
) -> Result<BTreeSet<String>, Gwt1EndToEndErrorV1> {
    validate_output_map(
        format!("perturbation {} baseline", perturbation.id),
        &perturbation.baseline_outputs,
    )?;
    validate_output_map(
        format!("perturbation {} perturbed", perturbation.id),
        &perturbation.perturbed_outputs,
    )?;

    Ok(GWT1_SPECIALISTS_V1
        .iter()
        .filter(|id| {
            perturbation.baseline_outputs.get(**id) != perturbation.perturbed_outputs.get(**id)
        })
        .map(|id| (*id).to_string())
        .collect())
}

fn derive_receipt(
    identity: &Gwt1ExecutionIdentityV1,
    raw: &Gwt1RawObservationsV1,
) -> Result<Gwt1SpecialistQualificationReceiptV1, Gwt1EndToEndErrorV1> {
    validate_raw_contract_v1(raw)?;

    validate_output_map("solo outputs", &raw.solo_panel.solo_outputs)?;
    validate_output_map("panel outputs", &raw.solo_panel.panel_outputs)?;

    for step in &raw.trajectory {
        validate_output_map(
            format!("trajectory step {} sequential", step.step),
            &step.sequential_outputs,
        )?;
        validate_output_map(
            format!("trajectory step {} parallel", step.step),
            &step.parallel_outputs,
        )?;
    }

    let worker_keys: BTreeSet<String> = raw.concurrency.worker_names.keys().cloned().collect();
    if worker_keys != canonical_specialists() {
        return Err(Gwt1EndToEndErrorV1::RawOutputCoverageMismatch {
            context: "concurrency worker identities".to_string(),
            observed: worker_keys.into_iter().collect(),
        });
    }

    let perturbations = raw
        .perturbations
        .iter()
        .map(|observation| {
            Ok(Gwt1PerturbationObservationV1 {
                id: observation.id.clone(),
                field: observation.field.clone(),
                target_specialist: observation.target_specialist.clone(),
                baseline_value: observation.baseline_value,
                perturbed_value: observation.perturbed_value,
                changed_specialists: changed_specialists(observation)?,
            })
        })
        .collect::<Result<Vec<_>, Gwt1EndToEndErrorV1>>()?;

    let solo_panel_equal = GWT1_SPECIALISTS_V1
        .iter()
        .map(|id| {
            (
                (*id).to_string(),
                raw.solo_panel.solo_outputs.get(*id) == raw.solo_panel.panel_outputs.get(*id),
            )
        })
        .collect();

    let sequential_parallel_equal = GWT1_SPECIALISTS_V1
        .iter()
        .map(|id| {
            let equal = raw.trajectory.iter().all(|step| {
                step.sequential_outputs.get(*id) == step.parallel_outputs.get(*id)
            });
            ((*id).to_string(), equal)
        })
        .collect();

    let distinct_workers_observed = raw
        .concurrency
        .worker_names
        .values()
        .collect::<BTreeSet<_>>()
        .len() as u32;

    let specialists = IMPLEMENTATION_PATHS_V1
        .iter()
        .map(|(id, implementation_path)| Gwt1SpecialistIdentityV1 {
            id: (*id).to_string(),
            implementation_path: (*implementation_path).to_string(),
            source_blob_sha: identity
                .specialist_blob_shas
                .get(*id)
                .cloned()
                .unwrap_or_default(),
        })
        .collect();

    Ok(Gwt1SpecialistQualificationReceiptV1 {
        schema: GWT1_QUALIFICATION_SCHEMA_V1.to_string(),
        source_commit_sha: identity.source_commit_sha.clone(),
        source_tree_sha: identity.source_tree_sha.clone(),
        execution_run_id: identity.execution_run_id.clone(),
        toolchain: identity.toolchain.clone(),
        specialists,
        perturbations,
        solo_panel_equal,
        trajectory_steps: raw.trajectory.len() as u32,
        sequential_parallel_equal,
        requested_workers: raw.concurrency.requested_workers,
        barrier_participants: raw.concurrency.barrier_participants,
        distinct_workers_observed,
        completed_specialists: raw.concurrency.completed_specialists.clone(),
        checkpoint_equal_supplementary: Some(raw.checkpoint_equal_supplementary),
    })
}

pub fn build_gwt1_evidence_v1(
    identity: &Gwt1ExecutionIdentityV1,
    raw: &Gwt1RawObservationsV1,
) -> Result<Gwt1EndToEndEvidenceV1, Gwt1EndToEndErrorV1> {
    let receipt = derive_receipt(identity, raw)?;
    let raw_observation_bytes = serde_json::to_vec(raw)
        .map_err(|error| Gwt1EndToEndErrorV1::RawSerialization(error.to_string()))?;
    let envelope = Gwt1EvidenceEnvelopeV1 {
        schema: GWT1_EVIDENCE_ENVELOPE_SCHEMA_V1.to_string(),
        raw_observations: describe_raw_observations_v1(&raw_observation_bytes),
        receipt,
    };
    let resolution = resolve_gwt1_evidence_envelope_v1(&envelope, &raw_observation_bytes);

    Ok(Gwt1EndToEndEvidenceV1 {
        raw_observation_bytes,
        envelope,
        resolution,
    })
}

pub fn run_gwt1_end_to_end_v1(
    identity: &Gwt1ExecutionIdentityV1,
) -> Result<Gwt1EndToEndEvidenceV1, Gwt1EndToEndErrorV1> {
    let raw = run_gwt1_specialist_qualification_v1()?;
    build_gwt1_evidence_v1(identity, &raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::butlin::gwt1_evidence_envelope::{
        Gwt1ArtifactIntegrityFailureV1, raw_observation_blake3,
    };
    use crate::benchmarks::butlin::gwt1_qualification::Gwt1QualificationOutcomeV1;

    fn identity() -> Gwt1ExecutionIdentityV1 {
        Gwt1ExecutionIdentityV1 {
            source_commit_sha: "a".repeat(40),
            source_tree_sha: "b".repeat(40),
            execution_run_id: "ci-run-test".to_string(),
            toolchain: "rustc-test".to_string(),
            specialist_blob_shas: GWT1_SPECIALISTS_V1
                .iter()
                .map(|id| ((*id).to_string(), "c".repeat(40)))
                .collect(),
        }
    }

    #[test]
    fn end_to_end_artifact_is_self_consistent() {
        let evidence = run_gwt1_end_to_end_v1(&identity()).expect("end-to-end GWT-1 run");
        assert_eq!(
            evidence.envelope.raw_observations.blake3,
            raw_observation_blake3(&evidence.raw_observation_bytes)
        );
        assert_eq!(
            evidence.envelope.raw_observations.byte_len,
            evidence.raw_observation_bytes.len() as u64
        );
        assert!(evidence.resolution.artifact_failures.is_empty());
    }

    #[test]
    fn tampered_raw_bytes_cannot_reuse_valid_summary() {
        let evidence = run_gwt1_end_to_end_v1(&identity()).expect("end-to-end GWT-1 run");
        let mut tampered = evidence.raw_observation_bytes.clone();
        tampered.push(b' ');
        let resolution = resolve_gwt1_evidence_envelope_v1(&evidence.envelope, &tampered);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Inconclusive);
        assert!(resolution.artifact_failures.iter().any(|failure| matches!(
            failure,
            Gwt1ArtifactIntegrityFailureV1::RawObservationDigestMismatch { .. }
        )));
    }

    #[test]
    fn missing_specialist_blob_identity_fails_closed() {
        let mut identity = identity();
        identity.specialist_blob_shas.remove("memory_manager");
        let evidence = run_gwt1_end_to_end_v1(&identity).expect("end-to-end GWT-1 run");
        assert_eq!(evidence.resolution.outcome, Gwt1QualificationOutcomeV1::Inconclusive);
    }

    #[test]
    fn wrong_contract_identity_is_rejected_before_receipt_construction() {
        let mut raw = run_gwt1_specialist_qualification_v1().expect("GWT-1 runner");
        raw.specialization_contract = "different-specialization-contract".to_string();
        assert!(matches!(
            build_gwt1_evidence_v1(&identity(), &raw),
            Err(Gwt1EndToEndErrorV1::RawContractMismatch { field, .. })
                if field == "specialization_contract"
        ));
    }

    #[test]
    fn reordered_or_duplicated_trajectory_step_is_rejected() {
        let mut raw = run_gwt1_specialist_qualification_v1().expect("GWT-1 runner");
        raw.trajectory[17].step = 16;
        assert!(matches!(
            build_gwt1_evidence_v1(&identity(), &raw),
            Err(Gwt1EndToEndErrorV1::RawTrajectoryStepMismatch {
                index: 17,
                observed: 16,
                expected: 17,
            })
        ));
    }

    #[test]
    fn altered_perturbation_dose_is_rejected() {
        let mut raw = run_gwt1_specialist_qualification_v1().expect("GWT-1 runner");
        raw.perturbations
            .iter_mut()
            .find(|p| p.id == "memory-unified-psi")
            .expect("memory perturbation")
            .perturbed_value = 0.79;
        assert!(matches!(
            build_gwt1_evidence_v1(&identity(), &raw),
            Err(Gwt1EndToEndErrorV1::RawPerturbationContractMismatch {
                perturbation_id,
                field,
            }) if perturbation_id == "memory-unified-psi" && field == "perturbed_value"
        ));
    }
}
