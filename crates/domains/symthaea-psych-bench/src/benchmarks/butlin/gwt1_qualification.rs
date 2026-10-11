// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance-bound direct qualification contract for Butlin GWT-1.
//!
//! This module is deliberately separate from `BehavioralIndicatorSignals`.
//! GWT-1's historical `specialization_fraction` is derived from other Butlin
//! signals and therefore cannot serve as an independent evidence lineage.
//!
//! A [`Gwt1SpecialistQualificationReceiptV1`] contains raw qualification
//! observations only. Callers cannot set a "passed" bit; [`resolve_gwt1_v1`]
//! derives a typed outcome fail-closed from the frozen V1 contract.
//!
//! `Qualified` here means only that this direct specialist-independence
//! qualification executed successfully. It is not a Butlin support tier and
//! does not imply consciousness, GWT-2/3/4, or global broadcast.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const GWT1_QUALIFICATION_SCHEMA_V1: &str = "butlin-gwt1-specialist-independence-v1";
pub const GWT1_MIN_TRAJECTORY_STEPS_V1: u32 = 48;
pub const GWT1_REQUIRED_WORKERS_V1: u32 = 4;

pub const GWT1_SPECIALISTS_V1: [&str; 4] = [
    "drive_manager",
    "memory_manager",
    "learning_manager",
    "perception_manager",
];

pub const GWT1_PERTURBATIONS_V1: [(&str, &str, &str); 4] = [
    ("drive-valence", "valence", "drive_manager"),
    ("memory-unified-psi", "unified_psi", "memory_manager"),
    (
        "learning-dissipative-health",
        "dissipative_health",
        "learning_manager",
    ),
    (
        "perception-phenomenal-binding",
        "phenomenal_binding",
        "perception_manager",
    ),
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1SpecialistIdentityV1 {
    pub id: String,
    pub implementation_path: String,
    pub source_blob_sha: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gwt1PerturbationObservationV1 {
    pub id: String,
    pub field: String,
    pub target_specialist: String,
    pub baseline_value: f64,
    pub perturbed_value: f64,
    pub changed_specialists: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gwt1SpecialistQualificationReceiptV1 {
    pub schema: String,
    pub source_commit_sha: String,
    pub source_tree_sha: String,
    pub execution_run_id: String,
    pub toolchain: String,
    pub specialists: Vec<Gwt1SpecialistIdentityV1>,
    pub perturbations: Vec<Gwt1PerturbationObservationV1>,
    pub solo_panel_equal: BTreeMap<String, bool>,
    pub trajectory_steps: u32,
    pub sequential_parallel_equal: BTreeMap<String, bool>,
    pub requested_workers: u32,
    pub barrier_participants: u32,
    pub distinct_workers_observed: u32,
    pub completed_specialists: BTreeSet<String>,
    /// Supplementary only until manager checkpoint replay-completeness is proven.
    pub checkpoint_equal_supplementary: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gwt1QualificationOutcomeV1 {
    Qualified,
    NotDemonstrated,
    Contradicted,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Gwt1QualificationFailureV1 {
    MissingExecutionIdentity { field: String },
    InvalidSha { field: String, value: String },
    SpecialistSetMismatch { observed: Vec<String> },
    SpecialistIdentityInvalid { specialist: String, field: String },
    PerturbationSetMismatch { observed: Vec<String> },
    PerturbationContractInvalid { perturbation_id: String, field: String },
    ObservationReferencesUnknownSpecialist {
        perturbation_id: String,
        specialist: String,
    },
    SoloPanelCoverageMismatch { observed: Vec<String> },
    SequentialParallelCoverageMismatch { observed: Vec<String> },
    TrajectoryTooShort { observed: u32, minimum: u32 },
    ConcurrencyProtocolInvalid {
        requested_workers: u32,
        barrier_participants: u32,
    },
    IncompleteConcurrentCompletion { observed: Vec<String> },
    ImpossibleWorkerCount {
        distinct_workers: u32,
        requested_workers: u32,
    },
    TargetNotResponsive {
        perturbation_id: String,
        target_specialist: String,
    },
    CrossSpecialistLeakage {
        perturbation_id: String,
        leaked_specialists: Vec<String>,
    },
    SoloPanelMismatch { specialist: String },
    SequentialParallelMismatch { specialist: String },
    ConcurrencyNotEstablished { observed: u32, required: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1QualificationResolutionV1 {
    pub outcome: Gwt1QualificationOutcomeV1,
    pub failures: Vec<Gwt1QualificationFailureV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FailureClass {
    NotDemonstrated,
    Contradicted,
    Inconclusive,
}

fn canonical_specialists() -> BTreeSet<String> {
    GWT1_SPECIALISTS_V1
        .iter()
        .map(|id| (*id).to_string())
        .collect()
}

fn valid_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn classify_failure(failure: &Gwt1QualificationFailureV1) -> FailureClass {
    use Gwt1QualificationFailureV1::*;
    match failure {
        TargetNotResponsive { .. } | ConcurrencyNotEstablished { .. } => {
            FailureClass::NotDemonstrated
        }
        CrossSpecialistLeakage { .. }
        | SoloPanelMismatch { .. }
        | SequentialParallelMismatch { .. } => FailureClass::Contradicted,
        MissingExecutionIdentity { .. }
        | InvalidSha { .. }
        | SpecialistSetMismatch { .. }
        | SpecialistIdentityInvalid { .. }
        | PerturbationSetMismatch { .. }
        | PerturbationContractInvalid { .. }
        | ObservationReferencesUnknownSpecialist { .. }
        | SoloPanelCoverageMismatch { .. }
        | SequentialParallelCoverageMismatch { .. }
        | TrajectoryTooShort { .. }
        | ConcurrencyProtocolInvalid { .. }
        | IncompleteConcurrentCompletion { .. }
        | ImpossibleWorkerCount { .. } => FailureClass::Inconclusive,
    }
}

pub fn resolve_gwt1_v1(
    receipt: &Gwt1SpecialistQualificationReceiptV1,
) -> Gwt1QualificationResolutionV1 {
    use Gwt1QualificationFailureV1::*;

    let canonical = canonical_specialists();
    let mut failures = Vec::new();

    if receipt.schema != GWT1_QUALIFICATION_SCHEMA_V1 {
        failures.push(PerturbationContractInvalid {
            perturbation_id: "receipt".to_string(),
            field: "schema".to_string(),
        });
    }

    for (field, value) in [
        ("execution_run_id", receipt.execution_run_id.as_str()),
        ("toolchain", receipt.toolchain.as_str()),
    ] {
        if value.trim().is_empty() {
            failures.push(MissingExecutionIdentity {
                field: field.to_string(),
            });
        }
    }

    for (field, value) in [
        ("source_commit_sha", receipt.source_commit_sha.as_str()),
        ("source_tree_sha", receipt.source_tree_sha.as_str()),
    ] {
        if !valid_git_sha(value) {
            failures.push(InvalidSha {
                field: field.to_string(),
                value: value.to_string(),
            });
        }
    }

    let specialist_ids: BTreeSet<String> =
        receipt.specialists.iter().map(|s| s.id.clone()).collect();
    if specialist_ids != canonical || receipt.specialists.len() != canonical.len() {
        failures.push(SpecialistSetMismatch {
            observed: receipt.specialists.iter().map(|s| s.id.clone()).collect(),
        });
    }

    for specialist in &receipt.specialists {
        if specialist.implementation_path.trim().is_empty() {
            failures.push(SpecialistIdentityInvalid {
                specialist: specialist.id.clone(),
                field: "implementation_path".to_string(),
            });
        }
        if !valid_git_sha(&specialist.source_blob_sha) {
            failures.push(SpecialistIdentityInvalid {
                specialist: specialist.id.clone(),
                field: "source_blob_sha".to_string(),
            });
        }
    }

    let perturbation_ids: BTreeSet<String> =
        receipt.perturbations.iter().map(|p| p.id.clone()).collect();
    let expected_perturbation_ids: BTreeSet<String> = GWT1_PERTURBATIONS_V1
        .iter()
        .map(|(id, _, _)| (*id).to_string())
        .collect();
    if perturbation_ids != expected_perturbation_ids
        || receipt.perturbations.len() != expected_perturbation_ids.len()
    {
        failures.push(PerturbationSetMismatch {
            observed: receipt.perturbations.iter().map(|p| p.id.clone()).collect(),
        });
    }

    for (expected_id, expected_field, expected_target) in GWT1_PERTURBATIONS_V1 {
        let Some(observation) = receipt
            .perturbations
            .iter()
            .find(|observation| observation.id == expected_id)
        else {
            continue;
        };

        if observation.field != expected_field {
            failures.push(PerturbationContractInvalid {
                perturbation_id: expected_id.to_string(),
                field: "field".to_string(),
            });
        }
        if observation.target_specialist != expected_target {
            failures.push(PerturbationContractInvalid {
                perturbation_id: expected_id.to_string(),
                field: "target_specialist".to_string(),
            });
        }
        if !observation.baseline_value.is_finite()
            || !observation.perturbed_value.is_finite()
            || observation.baseline_value.to_bits() == observation.perturbed_value.to_bits()
        {
            failures.push(PerturbationContractInvalid {
                perturbation_id: expected_id.to_string(),
                field: "values".to_string(),
            });
        }
        for specialist in &observation.changed_specialists {
            if !canonical.contains(specialist) {
                failures.push(ObservationReferencesUnknownSpecialist {
                    perturbation_id: expected_id.to_string(),
                    specialist: specialist.clone(),
                });
            }
        }
    }

    let solo_keys: BTreeSet<String> = receipt.solo_panel_equal.keys().cloned().collect();
    if solo_keys != canonical {
        failures.push(SoloPanelCoverageMismatch {
            observed: solo_keys.into_iter().collect(),
        });
    }

    let parallel_keys: BTreeSet<String> =
        receipt.sequential_parallel_equal.keys().cloned().collect();
    if parallel_keys != canonical {
        failures.push(SequentialParallelCoverageMismatch {
            observed: parallel_keys.into_iter().collect(),
        });
    }

    if receipt.trajectory_steps < GWT1_MIN_TRAJECTORY_STEPS_V1 {
        failures.push(TrajectoryTooShort {
            observed: receipt.trajectory_steps,
            minimum: GWT1_MIN_TRAJECTORY_STEPS_V1,
        });
    }

    if receipt.requested_workers != GWT1_REQUIRED_WORKERS_V1
        || receipt.barrier_participants != GWT1_REQUIRED_WORKERS_V1
    {
        failures.push(ConcurrencyProtocolInvalid {
            requested_workers: receipt.requested_workers,
            barrier_participants: receipt.barrier_participants,
        });
    }

    if receipt.completed_specialists != canonical {
        failures.push(IncompleteConcurrentCompletion {
            observed: receipt.completed_specialists.iter().cloned().collect(),
        });
    }

    if receipt.distinct_workers_observed > receipt.requested_workers {
        failures.push(ImpossibleWorkerCount {
            distinct_workers: receipt.distinct_workers_observed,
            requested_workers: receipt.requested_workers,
        });
    }

    let has_protocol_failure = failures
        .iter()
        .any(|failure| classify_failure(failure) == FailureClass::Inconclusive);

    if !has_protocol_failure {
        for (expected_id, _, expected_target) in GWT1_PERTURBATIONS_V1 {
            let observation = receipt
                .perturbations
                .iter()
                .find(|observation| observation.id == expected_id)
                .expect("validated perturbation coverage");

            if !observation.changed_specialists.contains(expected_target) {
                failures.push(TargetNotResponsive {
                    perturbation_id: expected_id.to_string(),
                    target_specialist: expected_target.to_string(),
                });
            }

            let leaked_specialists: Vec<String> = observation
                .changed_specialists
                .iter()
                .filter(|specialist| specialist.as_str() != expected_target)
                .cloned()
                .collect();
            if !leaked_specialists.is_empty() {
                failures.push(CrossSpecialistLeakage {
                    perturbation_id: expected_id.to_string(),
                    leaked_specialists,
                });
            }
        }

        for specialist in GWT1_SPECIALISTS_V1 {
            if receipt.solo_panel_equal.get(specialist) == Some(&false) {
                failures.push(SoloPanelMismatch {
                    specialist: specialist.to_string(),
                });
            }
            if receipt.sequential_parallel_equal.get(specialist) == Some(&false) {
                failures.push(SequentialParallelMismatch {
                    specialist: specialist.to_string(),
                });
            }
        }

        if receipt.distinct_workers_observed < GWT1_REQUIRED_WORKERS_V1 {
            failures.push(ConcurrencyNotEstablished {
                observed: receipt.distinct_workers_observed,
                required: GWT1_REQUIRED_WORKERS_V1,
            });
        }
    }

    let worst = failures.iter().map(classify_failure).max();
    let outcome = match worst {
        None => Gwt1QualificationOutcomeV1::Qualified,
        Some(FailureClass::NotDemonstrated) => Gwt1QualificationOutcomeV1::NotDemonstrated,
        Some(FailureClass::Contradicted) => Gwt1QualificationOutcomeV1::Contradicted,
        Some(FailureClass::Inconclusive) => Gwt1QualificationOutcomeV1::Inconclusive,
    };

    Gwt1QualificationResolutionV1 { outcome, failures }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_all(value: bool) -> BTreeMap<String, bool> {
        GWT1_SPECIALISTS_V1
            .iter()
            .map(|id| ((*id).to_string(), value))
            .collect()
    }

    fn canonical_receipt() -> Gwt1SpecialistQualificationReceiptV1 {
        let specialists = GWT1_SPECIALISTS_V1
            .iter()
            .map(|id| Gwt1SpecialistIdentityV1 {
                id: (*id).to_string(),
                implementation_path: format!("src/cognitive_loop/managers/{id}.rs"),
                source_blob_sha: "b".repeat(40),
            })
            .collect();

        let perturbations = GWT1_PERTURBATIONS_V1
            .iter()
            .map(|(id, field, target)| Gwt1PerturbationObservationV1 {
                id: (*id).to_string(),
                field: (*field).to_string(),
                target_specialist: (*target).to_string(),
                baseline_value: 0.4,
                perturbed_value: 0.8,
                changed_specialists: BTreeSet::from([(*target).to_string()]),
            })
            .collect();

        Gwt1SpecialistQualificationReceiptV1 {
            schema: GWT1_QUALIFICATION_SCHEMA_V1.to_string(),
            source_commit_sha: "a".repeat(40),
            source_tree_sha: "c".repeat(40),
            execution_run_id: "ci-run-123".to_string(),
            toolchain: "rustc 1.94.0".to_string(),
            specialists,
            perturbations,
            solo_panel_equal: map_all(true),
            trajectory_steps: GWT1_MIN_TRAJECTORY_STEPS_V1,
            sequential_parallel_equal: map_all(true),
            requested_workers: GWT1_REQUIRED_WORKERS_V1,
            barrier_participants: GWT1_REQUIRED_WORKERS_V1,
            distinct_workers_observed: GWT1_REQUIRED_WORKERS_V1,
            completed_specialists: canonical_specialists(),
            checkpoint_equal_supplementary: Some(true),
        }
    }

    #[test]
    fn canonical_receipt_resolves_qualified() {
        let resolution = resolve_gwt1_v1(&canonical_receipt());
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Qualified);
        assert!(resolution.failures.is_empty());
    }

    #[test]
    fn caller_cannot_hide_target_nonresponse() {
        let mut receipt = canonical_receipt();
        receipt.perturbations[0].changed_specialists.clear();

        let resolution = resolve_gwt1_v1(&receipt);
        assert_eq!(
            resolution.outcome,
            Gwt1QualificationOutcomeV1::NotDemonstrated
        );
        assert!(resolution.failures.iter().any(|failure| matches!(
            failure,
            Gwt1QualificationFailureV1::TargetNotResponsive { .. }
        )));
    }

    #[test]
    fn cross_specialist_leakage_is_contradicted() {
        let mut receipt = canonical_receipt();
        receipt.perturbations[0]
            .changed_specialists
            .insert("memory_manager".to_string());

        let resolution = resolve_gwt1_v1(&receipt);
        assert_eq!(
            resolution.outcome,
            Gwt1QualificationOutcomeV1::Contradicted
        );
    }

    #[test]
    fn missing_execution_identity_is_inconclusive() {
        let mut receipt = canonical_receipt();
        receipt.execution_run_id.clear();

        let resolution = resolve_gwt1_v1(&receipt);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Inconclusive);
    }

    #[test]
    fn trajectory_shortfall_is_inconclusive() {
        let mut receipt = canonical_receipt();
        receipt.trajectory_steps = GWT1_MIN_TRAJECTORY_STEPS_V1 - 1;

        let resolution = resolve_gwt1_v1(&receipt);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Inconclusive);
    }

    #[test]
    fn parallel_mismatch_is_contradicted() {
        let mut receipt = canonical_receipt();
        receipt
            .sequential_parallel_equal
            .insert("learning_manager".to_string(), false);

        let resolution = resolve_gwt1_v1(&receipt);
        assert_eq!(
            resolution.outcome,
            Gwt1QualificationOutcomeV1::Contradicted
        );
    }

    #[test]
    fn insufficient_workers_is_not_demonstrated() {
        let mut receipt = canonical_receipt();
        receipt.distinct_workers_observed = 3;

        let resolution = resolve_gwt1_v1(&receipt);
        assert_eq!(
            resolution.outcome,
            Gwt1QualificationOutcomeV1::NotDemonstrated
        );
    }

    #[test]
    fn json_roundtrip_preserves_receipt_and_resolution() {
        let receipt = canonical_receipt();
        let json = serde_json::to_string(&receipt).expect("serialize receipt");
        let decoded: Gwt1SpecialistQualificationReceiptV1 =
            serde_json::from_str(&json).expect("deserialize receipt");
        assert_eq!(decoded, receipt);
        assert_eq!(resolve_gwt1_v1(&decoded), resolve_gwt1_v1(&receipt));
    }

    #[test]
    fn duplicate_specialist_identity_is_inconclusive() {
        let mut receipt = canonical_receipt();
        receipt.specialists[3].id = "drive_manager".to_string();

        let resolution = resolve_gwt1_v1(&receipt);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Inconclusive);
    }
}
