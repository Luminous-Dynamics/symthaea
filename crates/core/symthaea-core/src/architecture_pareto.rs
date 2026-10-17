// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Report-only Pareto comparison for task-conditioned exosuit architecture research.
//!
//! The evaluator exposes trade-offs rather than producing a scalar "best suit" score.
//! Inputs are caller-declared measurements or synthetic values; the evaluator cannot
//! verify sensor calibration, clinical benefit, physical safety or actual task completion.
//! It emits no actuator commands and is not a hardware controller.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Current schema version for architecture comparison requests and reports.
pub const ARCHITECTURE_PARETO_SCHEMA_VERSION: u16 = 1;
/// Tolerance for segment-mass balance and total-system-mass consistency, in kilograms.
pub const MASS_BALANCE_TOLERANCE_KG_V1: f64 = 0.01;
/// Small tolerance for floating-point Pareto comparisons.
pub const PARETO_EPSILON_V1: f64 = 1.0e-9;

/// Evidence level supplied by the caller. Evaluation never promotes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureEvidenceLevelV1 {
    /// Synthetic input/replay only.
    SyntheticReplay,
    /// Deterministic simulation with a declared model.
    Simulation,
    /// Bench measurement on a physical test setup.
    Bench,
    /// Hardware-in-the-loop experiment.
    HardwareInLoop,
    /// Data collected in an appropriately reviewed human study.
    HumanStudy,
}

/// High-level architecture family; this is a descriptor, not a ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureFamilyV1 {
    /// Passive spring, elastic or load-transfer system.
    Passive,
    /// Soft textile/cable exosuit.
    SoftExosuit,
    /// Rigid structure with locally worn powered actuation.
    PoweredRigid,
    /// Remote actuation with a tether/transmission to the wearable interface.
    RemoteActuationTethered,
    /// Combination of powered assistance and passive mechanical support.
    HybridPoweredPassive,
    /// A declared architecture not represented by the other categories.
    Other,
}

/// Operating-condition field referenced by context-validation findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureConditionFieldV1 {
    /// Declared speed in metres/second.
    SpeedMps,
    /// Payload in kilograms.
    PayloadKg,
    /// Ground grade in percent.
    GradePercent,
    /// Ambient temperature in degrees Celsius.
    AmbientTemperatureC,
}

/// Body-segment distribution of wearer-borne device mass, in kilograms.
///
/// The component values must sum to wearer_borne_mass_kg within
/// MASS_BALANCE_TOLERANCE_KG_V1. Values describe device mass, not human body mass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WearerMassDistributionV1 {
    /// Worn mass attached to torso/back.
    pub torso_kg: f64,
    /// Worn mass attached around pelvis/hips.
    pub pelvis_kg: f64,
    /// Worn mass near proximal limb segments.
    pub proximal_limb_kg: f64,
    /// Worn mass near distal limb segments.
    pub distal_limb_kg: f64,
    /// Other mass worn by the user.
    pub other_worn_kg: f64,
}

/// Task-conditioned dimensions for a single candidate architecture.
///
/// Every numeric dimension is required. Missing values must not be encoded as zero:
/// missing JSON fields fail deserialization and invalid numeric values fail evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureMetricsV1 {
    /// Eligible tasks meeting the declared success predicate divided by all eligible starts, [0, 1].
    pub task_success_rate: f64,
    /// All mass physically borne by the wearer, kg.
    pub wearer_borne_mass_kg: f64,
    /// Full system mass: worn, off-board equipment, transmission and declared parts, kg.
    pub total_system_mass_kg: f64,
    /// Off-board mass needed to operate/support this architecture, kg.
    pub offboard_mass_kg: f64,
    /// Distribution of wearer-borne mass by body segment, kg.
    pub mass_distribution: WearerMassDistributionV1,
    /// Electrical energy per declared task, Wh.
    pub energy_wh_per_task: f64,
    /// Setup time to reach the declared ready state, seconds.
    pub setup_time_seconds: f64,
    /// Declared movement-restriction index, [0, 1], with higher meaning more restriction.
    pub movement_restriction_index: f64,
    /// Comfort score on a declared normalized scale [0, 1], with higher meaning more comfortable.
    ///
    /// This is a supplied score/proxy, not a clinical finding.
    pub comfort_score: f64,
    /// Tether-related interference events per task.
    ///
    /// A missing measurement must not be imputed as zero. A declared untethered
    /// design may report a structural zero for this metric.
    pub tether_interference_events_per_task: f64,
    /// Time to complete a declared repair/replacement task, minutes.
    pub repair_time_minutes: f64,
    /// Number of service-relevant components in the declared bill of materials.
    pub service_component_count: u32,
}

/// One candidate architecture evaluated on common task/profile/environment conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureCandidateV1 {
    /// Stable unique identifier within a comparison set.
    pub candidate_id: String,
    /// High-level architecture family.
    pub family: ArchitectureFamilyV1,
    /// Whether the wearer/system relies on a physical tether during the declared task.
    pub tethered: bool,
    /// Declared candidate metrics under the comparison request's common conditions.
    pub metrics: ArchitectureMetricsV1,
}

/// Operating conditions common to every candidate in a comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureOperatingConditionsV1 {
    /// Declared speed, metres/second.
    pub speed_mps: f64,
    /// Payload carried/handled for this task, kg.
    pub payload_kg: f64,
    /// Ground grade in percent; negative values may represent descent.
    pub grade_percent: f64,
    /// Ambient temperature, degrees Celsius.
    pub ambient_temperature_c: f64,
}

/// Versioned request for a fair, task-conditioned architecture comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureComparisonRequestV1 {
    /// Must equal ARCHITECTURE_PARETO_SCHEMA_VERSION.
    pub schema_version: u16,
    /// Stable benchmark scenario identifier.
    pub benchmark_case_id: String,
    /// Stable declared task identifier.
    pub task_id: String,
    /// Wearer/device profile identifier.
    pub wearer_profile_id: String,
    /// Environment profile identifier.
    pub environment_profile_id: String,
    /// Task protocol revision; comparisons across versions are invalid.
    pub task_protocol_version: String,
    /// Declared evidence level. The evaluator never promotes this value.
    pub evidence_level: ArchitectureEvidenceLevelV1,
    /// Identical declared operating conditions for all candidates.
    pub operating_conditions: ArchitectureOperatingConditionsV1,
    /// Candidate architectures, kept in caller input order in the report.
    pub candidates: Vec<ArchitectureCandidateV1>,
}

/// Metric objective included in a dominance explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureMetricV1 {
    /// Higher is better.
    TaskSuccessRate,
    /// Lower is better.
    WearerBorneMassKg,
    /// Lower is better.
    TotalSystemMassKg,
    /// Lower is better.
    OffboardMassKg,
    /// Lower is better.
    TorsoMassKg,
    /// Lower is better.
    PelvisMassKg,
    /// Lower is better.
    ProximalLimbMassKg,
    /// Lower is better.
    DistalLimbMassKg,
    /// Lower is better.
    OtherWornMassKg,
    /// Lower is better.
    EnergyWhPerTask,
    /// Lower is better.
    SetupTimeSeconds,
    /// Lower is better.
    MovementRestrictionIndex,
    /// Higher is better.
    ComfortScore,
    /// Lower is better.
    TetherInterferenceEventsPerTask,
    /// Lower is better.
    RepairTimeMinutes,
    /// Lower is better.
    ServiceComponentCount,
}

/// Why a candidate cannot be included in a meaningful comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureValidationFindingV1 {
    /// Request uses an unsupported schema version.
    UnsupportedSchemaVersion { received: u16 },
    /// Required comparison identity is empty or contains surrounding whitespace.
    InvalidComparisonIdentity { field: String },
    /// Operating-condition value is not finite.
    NonFiniteOperatingCondition { field: ArchitectureConditionFieldV1 },
    /// Operating-condition value is negative where not permitted.
    NegativeOperatingCondition { field: ArchitectureConditionFieldV1 },
    /// Fewer than two candidate architectures were provided.
    InsufficientCandidates,
    /// Candidate IDs must be unique and non-empty.
    InvalidCandidateId { candidate_id: String },
    /// One candidate ID appears more than once.
    DuplicateCandidateId { candidate_id: String },
    /// A numeric dimension is not finite.
    NonFiniteMetric { candidate_id: String, metric: ArchitectureMetricV1 },
    /// A dimension expected to be non-negative was negative.
    NegativeMetric { candidate_id: String, metric: ArchitectureMetricV1 },
    /// A normalized dimension fell outside [0, 1].
    OutsideNormalizedRange { candidate_id: String, metric: ArchitectureMetricV1 },
    /// Segment mass distribution does not reconcile with declared wearer-borne mass.
    MassDistributionMismatch {
        candidate_id: String,
        segment_sum_kg: f64,
        declared_wearer_mass_kg: f64,
    },
    /// Aggregating individually finite mass values overflowed.
    MassAggregationOverflow { candidate_id: String },
    /// Total-system mass understates wearer mass plus off-board mass.
    TotalMassInconsistent {
        candidate_id: String,
        declared_total_kg: f64,
        minimum_consistent_total_kg: f64,
    },
    /// A tether interference count was supplied for a declared non-tethered architecture.
    TetherMetricInconsistent { candidate_id: String },
    /// A remote-actuation/tethered architecture was declared non-tethered.
    RemoteActuationRequiresTether { candidate_id: String },
}

/// A candidate proven to dominate another within the declared metric set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureDominanceV1 {
    /// Candidate that dominates the assessed candidate.
    pub dominating_candidate_id: String,
    /// Dimensions on which the dominating candidate is strictly better.
    pub improved_dimensions: Vec<ArchitectureMetricV1>,
}

/// Per-candidate assessment. dominated_by remains empty for Pareto-optimal candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureCandidateAssessmentV1 {
    /// Candidate identifier from the request.
    pub candidate_id: String,
    /// Whether the candidate is on the non-dominated frontier.
    pub is_pareto_optimal: bool,
    /// All candidates that dominate this one, in original input order.
    pub dominated_by: Vec<ArchitectureDominanceV1>,
}

/// Report-only Pareto frontier result; no scalar aggregate score is produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureParetoReportV1 {
    /// Report schema version.
    pub schema_version: u16,
    /// Benchmark scenario identifier.
    pub benchmark_case_id: String,
    /// Task identity shared by the comparison.
    pub task_id: String,
    /// Wearer profile shared by the comparison.
    pub wearer_profile_id: String,
    /// Environment profile shared by the comparison.
    pub environment_profile_id: String,
    /// Task protocol version shared by the comparison.
    pub task_protocol_version: String,
    /// Caller-declared evidence level; never auto-promoted.
    pub evidence_level: ArchitectureEvidenceLevelV1,
    /// Assessments preserve input order.
    pub assessments: Vec<ArchitectureCandidateAssessmentV1>,
    /// Explicit reminder of what this evaluator can and cannot claim.
    pub claim_ceiling: String,
}

/// Evaluate a comparison set and return the non-dominated Pareto frontier.
///
/// No weighted score is constructed. The request fixes one task/profile/environment
/// and protocol context so candidates are comparable. Dominance uses every declared
/// metric, including wearer and off-board mass, segment mass, energy, setup, restriction,
/// comfort, tether interference and repair complexity. A missing metric fails strict
/// deserialization rather than being filled with zero. Supplied values are not authenticated.
pub fn evaluate_architecture_pareto(
    request: &ArchitectureComparisonRequestV1,
) -> Result<ArchitectureParetoReportV1, Vec<ArchitectureValidationFindingV1>> {
    let mut findings = Vec::new();

    if request.schema_version != ARCHITECTURE_PARETO_SCHEMA_VERSION {
        findings.push(ArchitectureValidationFindingV1::UnsupportedSchemaVersion {
            received: request.schema_version,
        });
    }
    for (field, value) in [
        ("benchmark_case_id", request.benchmark_case_id.as_str()),
        ("task_id", request.task_id.as_str()),
        ("wearer_profile_id", request.wearer_profile_id.as_str()),
        ("environment_profile_id", request.environment_profile_id.as_str()),
        ("task_protocol_version", request.task_protocol_version.as_str()),
    ] {
        if value.trim().is_empty() || value.trim() != value {
            findings.push(ArchitectureValidationFindingV1::InvalidComparisonIdentity {
                field: field.to_owned(),
            });
        }
    }

    let conditions = &request.operating_conditions;
    for (field, value) in [
        (ArchitectureConditionFieldV1::SpeedMps, conditions.speed_mps),
        (ArchitectureConditionFieldV1::PayloadKg, conditions.payload_kg),
        (ArchitectureConditionFieldV1::GradePercent, conditions.grade_percent),
        (ArchitectureConditionFieldV1::AmbientTemperatureC, conditions.ambient_temperature_c),
    ] {
        if !value.is_finite() {
            findings.push(ArchitectureValidationFindingV1::NonFiniteOperatingCondition { field });
        } else if matches!(field, ArchitectureConditionFieldV1::SpeedMps | ArchitectureConditionFieldV1::PayloadKg)
            && value < 0.0
        {
            findings.push(ArchitectureValidationFindingV1::NegativeOperatingCondition { field });
        }
    }
    if request.candidates.len() < 2 {
        findings.push(ArchitectureValidationFindingV1::InsufficientCandidates);
    }

    let mut seen = HashSet::with_capacity(request.candidates.len());
    for candidate in &request.candidates {
        let id = candidate.candidate_id.as_str();
        if id.trim().is_empty() || id.trim() != id {
            findings.push(ArchitectureValidationFindingV1::InvalidCandidateId {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if !seen.insert(id) {
            findings.push(ArchitectureValidationFindingV1::DuplicateCandidateId {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if candidate.family == ArchitectureFamilyV1::RemoteActuationTethered && !candidate.tethered {
            findings.push(ArchitectureValidationFindingV1::RemoteActuationRequiresTether {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        findings.extend(validate_candidate_metrics(candidate));
    }

    if !findings.is_empty() {
        return Err(findings);
    }

    let mut assessments = Vec::with_capacity(request.candidates.len());
    for (index, candidate) in request.candidates.iter().enumerate() {
        let mut dominated_by = Vec::new();
        for (other_index, other) in request.candidates.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let improved_dimensions = dominance_dimensions(&other.metrics, &candidate.metrics);
            if !improved_dimensions.is_empty() {
                dominated_by.push(ArchitectureDominanceV1 {
                    dominating_candidate_id: other.candidate_id.clone(),
                    improved_dimensions,
                });
            }
        }
        assessments.push(ArchitectureCandidateAssessmentV1 {
            candidate_id: candidate.candidate_id.clone(),
            is_pareto_optimal: dominated_by.is_empty(),
            dominated_by,
        });
    }

    Ok(ArchitectureParetoReportV1 {
        schema_version: ARCHITECTURE_PARETO_SCHEMA_VERSION,
        benchmark_case_id: request.benchmark_case_id.clone(),
        task_id: request.task_id.clone(),
        wearer_profile_id: request.wearer_profile_id.clone(),
        environment_profile_id: request.environment_profile_id.clone(),
        task_protocol_version: request.task_protocol_version.clone(),
        evidence_level: request.evidence_level,
        assessments,
        claim_ceiling: "Report-only Pareto comparison. Supplied metrics and evidence level are not authenticated; no clinical benefit, physical safety, regulatory compliance, or hardware readiness is established.".to_owned(),
    })
}

fn validate_candidate_metrics(
    candidate: &ArchitectureCandidateV1,
) -> Vec<ArchitectureValidationFindingV1> {
    let id = &candidate.candidate_id;
    let m = &candidate.metrics;
    let mut findings = Vec::new();

    let non_negative = [
        (ArchitectureMetricV1::WearerBorneMassKg, m.wearer_borne_mass_kg),
        (ArchitectureMetricV1::TotalSystemMassKg, m.total_system_mass_kg),
        (ArchitectureMetricV1::OffboardMassKg, m.offboard_mass_kg),
        (ArchitectureMetricV1::TorsoMassKg, m.mass_distribution.torso_kg),
        (ArchitectureMetricV1::PelvisMassKg, m.mass_distribution.pelvis_kg),
        (ArchitectureMetricV1::ProximalLimbMassKg, m.mass_distribution.proximal_limb_kg),
        (ArchitectureMetricV1::DistalLimbMassKg, m.mass_distribution.distal_limb_kg),
        (ArchitectureMetricV1::OtherWornMassKg, m.mass_distribution.other_worn_kg),
        (ArchitectureMetricV1::EnergyWhPerTask, m.energy_wh_per_task),
        (ArchitectureMetricV1::SetupTimeSeconds, m.setup_time_seconds),
        (ArchitectureMetricV1::TetherInterferenceEventsPerTask, m.tether_interference_events_per_task),
        (ArchitectureMetricV1::RepairTimeMinutes, m.repair_time_minutes),
    ];
    for (metric, value) in non_negative {
        if !value.is_finite() {
            findings.push(ArchitectureValidationFindingV1::NonFiniteMetric {
                candidate_id: id.clone(), metric,
            });
        } else if value < 0.0 {
            findings.push(ArchitectureValidationFindingV1::NegativeMetric {
                candidate_id: id.clone(), metric,
            });
        }
    }
    for (metric, value) in [
        (ArchitectureMetricV1::TaskSuccessRate, m.task_success_rate),
        (ArchitectureMetricV1::MovementRestrictionIndex, m.movement_restriction_index),
        (ArchitectureMetricV1::ComfortScore, m.comfort_score),
    ] {
        if !value.is_finite() {
            findings.push(ArchitectureValidationFindingV1::NonFiniteMetric {
                candidate_id: id.clone(), metric,
            });
        } else if !(0.0..=1.0).contains(&value) {
            findings.push(ArchitectureValidationFindingV1::OutsideNormalizedRange {
                candidate_id: id.clone(), metric,
            });
        }
    }

    let mass_components = [
        m.wearer_borne_mass_kg,
        m.offboard_mass_kg,
        m.total_system_mass_kg,
        m.mass_distribution.torso_kg,
        m.mass_distribution.pelvis_kg,
        m.mass_distribution.proximal_limb_kg,
        m.mass_distribution.distal_limb_kg,
        m.mass_distribution.other_worn_kg,
    ];
    let mass_components_finite = mass_components.iter().all(|value| value.is_finite());
    if mass_components_finite {
        let segment_sum = m.mass_distribution.torso_kg
            + m.mass_distribution.pelvis_kg
            + m.mass_distribution.proximal_limb_kg
            + m.mass_distribution.distal_limb_kg
            + m.mass_distribution.other_worn_kg;
        let minimum_total = m.wearer_borne_mass_kg + m.offboard_mass_kg;
        if !segment_sum.is_finite() || !minimum_total.is_finite() {
            findings.push(ArchitectureValidationFindingV1::MassAggregationOverflow {
                candidate_id: id.clone(),
            });
        } else {
            if (segment_sum - m.wearer_borne_mass_kg).abs() > MASS_BALANCE_TOLERANCE_KG_V1 {
                findings.push(ArchitectureValidationFindingV1::MassDistributionMismatch {
                    candidate_id: id.clone(),
                    segment_sum_kg: segment_sum,
                    declared_wearer_mass_kg: m.wearer_borne_mass_kg,
                });
            }
            if m.total_system_mass_kg + MASS_BALANCE_TOLERANCE_KG_V1 < minimum_total {
                findings.push(ArchitectureValidationFindingV1::TotalMassInconsistent {
                    candidate_id: id.clone(),
                    declared_total_kg: m.total_system_mass_kg,
                    minimum_consistent_total_kg: minimum_total,
                });
            }
        }
    }
    if !candidate.tethered && m.tether_interference_events_per_task > PARETO_EPSILON_V1 {
        findings.push(ArchitectureValidationFindingV1::TetherMetricInconsistent {
            candidate_id: id.clone(),
        });
    }
    findings
}

/// Return strict improvements if a is no worse than b on every objective;
/// an empty vector also means equal metrics or trade-offs.
fn dominance_dimensions(
    a: &ArchitectureMetricsV1,
    b: &ArchitectureMetricsV1,
) -> Vec<ArchitectureMetricV1> {
    use ArchitectureMetricV1 as M;
    let mut improvements = Vec::new();
    let mut no_worse_everywhere = true;

    macro_rules! compare_lower {
        ($metric:expr, $a:expr, $b:expr) => {{
            let a_value = $a;
            let b_value = $b;
            if a_value > b_value + PARETO_EPSILON_V1 {
                no_worse_everywhere = false;
            } else if a_value + PARETO_EPSILON_V1 < b_value {
                improvements.push($metric);
            }
        }};
    }
    macro_rules! compare_higher {
        ($metric:expr, $a:expr, $b:expr) => {{
            let a_value = $a;
            let b_value = $b;
            if a_value + PARETO_EPSILON_V1 < b_value {
                no_worse_everywhere = false;
            } else if a_value > b_value + PARETO_EPSILON_V1 {
                improvements.push($metric);
            }
        }};
    }

    compare_higher!(M::TaskSuccessRate, a.task_success_rate, b.task_success_rate);
    compare_lower!(M::WearerBorneMassKg, a.wearer_borne_mass_kg, b.wearer_borne_mass_kg);
    compare_lower!(M::TotalSystemMassKg, a.total_system_mass_kg, b.total_system_mass_kg);
    compare_lower!(M::OffboardMassKg, a.offboard_mass_kg, b.offboard_mass_kg);
    compare_lower!(M::TorsoMassKg, a.mass_distribution.torso_kg, b.mass_distribution.torso_kg);
    compare_lower!(M::PelvisMassKg, a.mass_distribution.pelvis_kg, b.mass_distribution.pelvis_kg);
    compare_lower!(M::ProximalLimbMassKg, a.mass_distribution.proximal_limb_kg, b.mass_distribution.proximal_limb_kg);
    compare_lower!(M::DistalLimbMassKg, a.mass_distribution.distal_limb_kg, b.mass_distribution.distal_limb_kg);
    compare_lower!(M::OtherWornMassKg, a.mass_distribution.other_worn_kg, b.mass_distribution.other_worn_kg);
    compare_lower!(M::EnergyWhPerTask, a.energy_wh_per_task, b.energy_wh_per_task);
    compare_lower!(M::SetupTimeSeconds, a.setup_time_seconds, b.setup_time_seconds);
    compare_lower!(M::MovementRestrictionIndex, a.movement_restriction_index, b.movement_restriction_index);
    compare_higher!(M::ComfortScore, a.comfort_score, b.comfort_score);
    compare_lower!(M::TetherInterferenceEventsPerTask, a.tether_interference_events_per_task, b.tether_interference_events_per_task);
    compare_lower!(M::RepairTimeMinutes, a.repair_time_minutes, b.repair_time_minutes);
    if a.service_component_count < b.service_component_count {
        improvements.push(M::ServiceComponentCount);
    } else if a.service_component_count > b.service_component_count {
        no_worse_everywhere = false;
    }

    if no_worse_everywhere && !improvements.is_empty() { improvements } else { Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, success: f64, worn: f64, energy: f64, comfort: f64) -> ArchitectureCandidateV1 {
        let remaining = worn - 0.4 - 0.5 - 0.7 - 0.2;
        ArchitectureCandidateV1 {
            candidate_id: id.to_owned(),
            family: ArchitectureFamilyV1::HybridPoweredPassive,
            tethered: false,
            metrics: ArchitectureMetricsV1 {
                task_success_rate: success,
                wearer_borne_mass_kg: worn,
                total_system_mass_kg: worn,
                offboard_mass_kg: 0.0,
                mass_distribution: WearerMassDistributionV1 {
                    torso_kg: 0.4,
                    pelvis_kg: 0.5,
                    proximal_limb_kg: 0.7,
                    distal_limb_kg: 0.2,
                    other_worn_kg: remaining,
                },
                energy_wh_per_task: energy,
                setup_time_seconds: 30.0,
                movement_restriction_index: 0.2,
                comfort_score: comfort,
                tether_interference_events_per_task: 0.0,
                repair_time_minutes: 20.0,
                service_component_count: 15,
            },
        }
    }

    fn request(candidates: Vec<ArchitectureCandidateV1>) -> ArchitectureComparisonRequestV1 {
        ArchitectureComparisonRequestV1 {
            schema_version: ARCHITECTURE_PARETO_SCHEMA_VERSION,
            benchmark_case_id: "synthetic-case-001".to_owned(),
            task_id: "overground-walk".to_owned(),
            wearer_profile_id: "profile-a".to_owned(),
            environment_profile_id: "flat-lab".to_owned(),
            task_protocol_version: "v1".to_owned(),
            evidence_level: ArchitectureEvidenceLevelV1::SyntheticReplay,
            operating_conditions: ArchitectureOperatingConditionsV1 {
                speed_mps: 0.8,
                payload_kg: 0.0,
                grade_percent: 0.0,
                ambient_temperature_c: 22.0,
            },
            candidates,
        }
    }

    #[test]
    fn preserves_trade_offs_instead_of_choosing_one_scalar_winner() {
        let report = evaluate_architecture_pareto(&request(vec![
            candidate("light-comfort-first", 0.78, 2.0, 0.2, 0.95),
            candidate("heavier-task-performance", 0.94, 4.0, 1.2, 0.78),
        ]))
        .expect("valid comparison");
        assert_eq!(report.assessments.len(), 2);
        assert!(report.assessments.iter().all(|assessment| assessment.is_pareto_optimal));
        assert!(report.assessments.iter().all(|assessment| assessment.dominated_by.is_empty()));
    }

    #[test]
    fn candidate_dominated_on_every_declared_dimension_is_explained() {
        let report = evaluate_architecture_pareto(&request(vec![
            candidate("compact", 0.95, 2.0, 0.5, 0.9),
            candidate("overbuilt", 0.80, 4.0, 1.5, 0.7),
        ]))
        .expect("valid comparison");
        let overbuilt = report.assessments.iter().find(|assessment| assessment.candidate_id == "overbuilt")
            .expect("overbuilt candidate remains present");
        assert!(!overbuilt.is_pareto_optimal);
        assert_eq!(overbuilt.dominated_by.len(), 1);
        assert_eq!(overbuilt.dominated_by[0].dominating_candidate_id, "compact");
        assert!(overbuilt.dominated_by[0].improved_dimensions.contains(&ArchitectureMetricV1::TaskSuccessRate));
    }

    #[test]
    fn offboard_mass_and_mass_distribution_cannot_be_hidden() {
        let mut inconsistent = candidate("mislabeled", 0.9, 3.0, 0.5, 0.8);
        inconsistent.metrics.offboard_mass_kg = 10.0;
        inconsistent.metrics.total_system_mass_kg = 4.0;
        let result = evaluate_architecture_pareto(&request(vec![
            candidate("valid-a", 0.8, 3.0, 0.7, 0.7), inconsistent,
        ]));
        assert!(matches!(
            result,
            Err(ref findings) if findings.iter().any(|finding| matches!(finding, ArchitectureValidationFindingV1::TotalMassInconsistent { .. }))
        ));
    }

    #[test]
    fn missing_metric_is_not_imputed_as_zero_by_strict_schema() {
        let serialized = serde_json::to_value(candidate("candidate-a", 0.9, 3.0, 0.5, 0.8))
            .expect("candidate serializes");
        let mut malformed = serialized;
        malformed["metrics"].as_object_mut().expect("metrics object").remove("energy_wh_per_task");
        assert!(serde_json::from_value::<ArchitectureCandidateV1>(malformed).is_err());
    }

    #[test]
    fn mismatched_mass_distribution_is_rejected() {
        let mut malformed = candidate("candidate-a", 0.9, 3.0, 0.5, 0.8);
        malformed.metrics.mass_distribution.distal_limb_kg += 0.5;
        let result = evaluate_architecture_pareto(&request(vec![
            candidate("candidate-b", 0.8, 3.0, 0.7, 0.7), malformed,
        ]));
        assert!(matches!(
            result,
            Err(ref findings) if findings.iter().any(|finding| matches!(finding, ArchitectureValidationFindingV1::MassDistributionMismatch { .. }))
        ));
    }

    #[test]
    fn reject_duplicate_candidate_ids_and_invalid_values() {
        let mut invalid = candidate("same", 0.9, 3.0, 0.5, 0.8);
        invalid.metrics.energy_wh_per_task = f64::NAN;
        let duplicate = candidate("same", 0.8, 3.0, 0.7, 0.7);
        let result = evaluate_architecture_pareto(&request(vec![invalid, duplicate]));
        assert!(matches!(
            result,
            Err(ref findings) if findings.iter().any(|finding| matches!(finding, ArchitectureValidationFindingV1::DuplicateCandidateId { .. }))
                && findings.iter().any(|finding| matches!(finding, ArchitectureValidationFindingV1::NonFiniteMetric { metric: ArchitectureMetricV1::EnergyWhPerTask, .. }))
        ));
    }

    #[test]
    fn mass_aggregation_overflow_fails_closed() {
        let mut huge = candidate("huge", 0.9, 3.0, 0.5, 0.8);
        huge.metrics.wearer_borne_mass_kg = f64::MAX * 0.6;
        huge.metrics.offboard_mass_kg = f64::MAX * 0.6;
        huge.metrics.total_system_mass_kg = f64::MAX;
        huge.metrics.mass_distribution = WearerMassDistributionV1 {
            torso_kg: f64::MAX * 0.6,
            pelvis_kg: 0.0,
            proximal_limb_kg: 0.0,
            distal_limb_kg: 0.0,
            other_worn_kg: 0.0,
        };
        let result = evaluate_architecture_pareto(&request(vec![
            huge,
            candidate("reference", 0.8, 3.0, 0.7, 0.7),
        ]));
        assert!(matches!(
            result,
            Err(ref findings) if findings.iter().any(|finding| matches!(
                finding,
                ArchitectureValidationFindingV1::MassAggregationOverflow { .. }
            ))
        ));
    }

    #[test]
    fn remote_actuation_must_declare_a_tether() {
        let mut remote = candidate("remote", 0.9, 3.0, 0.5, 0.8);
        remote.family = ArchitectureFamilyV1::RemoteActuationTethered;
        remote.tethered = false;
        let result = evaluate_architecture_pareto(&request(vec![
            remote,
            candidate("local", 0.8, 3.0, 0.7, 0.7),
        ]));
        assert!(matches!(
            result,
            Err(ref findings) if findings.iter().any(|finding| matches!(
                finding,
                ArchitectureValidationFindingV1::RemoteActuationRequiresTether { .. }
            ))
        ));
    }

    #[test]
    fn invalid_context_and_schema_cannot_produce_a_frontier() {
        let mut input = request(vec![
            candidate("candidate-a", 0.8, 2.5, 0.4, 0.9),
            candidate("candidate-b", 0.95, 4.0, 0.8, 0.7),
        ]);
        input.schema_version += 1;
        input.benchmark_case_id = " ".to_owned();
        input.operating_conditions.speed_mps = f64::NAN;
        let result = evaluate_architecture_pareto(&input);
        assert!(matches!(
            result,
            Err(ref findings)
                if findings.iter().any(|finding| matches!(finding, ArchitectureValidationFindingV1::UnsupportedSchemaVersion { .. }))
                && findings.iter().any(|finding| matches!(finding, ArchitectureValidationFindingV1::InvalidComparisonIdentity { .. }))
                && findings.iter().any(|finding| matches!(finding, ArchitectureValidationFindingV1::NonFiniteOperatingCondition { field: ArchitectureConditionFieldV1::SpeedMps }))
        ));
    }

    #[test]
    fn deterministic_reports_round_trip_to_identical_json() {
        let input = request(vec![
            candidate("candidate-a", 0.8, 2.5, 0.4, 0.9),
            candidate("candidate-b", 0.95, 4.0, 0.8, 0.7),
            candidate("candidate-c", 0.7, 5.0, 1.0, 0.65),
        ]);
        let expected = evaluate_architecture_pareto(&input).expect("valid comparison");
        let bytes = serde_json::to_vec(&expected).expect("report serializes");
        for _ in 0..100 {
            let actual = evaluate_architecture_pareto(&input).expect("valid comparison");
            assert_eq!(actual, expected);
            assert_eq!(serde_json::to_vec(&actual).expect("report serializes"), bytes);
        }
    }
}
