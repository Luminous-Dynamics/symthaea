// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conservative interval-aware Pareto screening for wearable-robot architecture research.
//!
//! This module complements `architecture_pareto`, which compares point estimates.
//! It accepts caller-declared conservative metric envelopes and distinguishes robust
//! dominance from comparisons that remain unresolved within those bounds. It does not
//! assign probabilities, infer correlations, authenticate measurements, or control hardware.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::architecture_pareto::{
    ArchitectureConditionFieldV1, ArchitectureEvidenceLevelV1, ArchitectureFamilyV1,
    ArchitectureMetricV1, ArchitectureOperatingConditionsV1,
};

/// Current schema version for interval-aware architecture comparisons.
pub const ARCHITECTURE_ROBUSTNESS_SCHEMA_VERSION_V1: u16 = 1;
/// Tolerance for interval mass-envelope consistency checks, in kilograms.
pub const ROBUST_MASS_BALANCE_TOLERANCE_KG_V1: f64 = 0.01;
/// Numeric comparison tolerance; not a measurement uncertainty allowance.
pub const ROBUST_COMPARISON_EPSILON_V1: f64 = 1.0e-9;

const REQUIRED_METRICS: [ArchitectureMetricV1; 15] = [
    ArchitectureMetricV1::TaskSuccessRate,
    ArchitectureMetricV1::WearerBorneMassKg,
    ArchitectureMetricV1::TotalSystemMassKg,
    ArchitectureMetricV1::OffboardMassKg,
    ArchitectureMetricV1::TorsoMassKg,
    ArchitectureMetricV1::PelvisMassKg,
    ArchitectureMetricV1::ProximalLimbMassKg,
    ArchitectureMetricV1::DistalLimbMassKg,
    ArchitectureMetricV1::OtherWornMassKg,
    ArchitectureMetricV1::EnergyWhPerTask,
    ArchitectureMetricV1::SetupTimeSeconds,
    ArchitectureMetricV1::MovementRestrictionIndex,
    ArchitectureMetricV1::ComfortScore,
    ArchitectureMetricV1::TetherInterferenceEventsPerTask,
    ArchitectureMetricV1::RepairTimeMinutes,
];

/// Interval meaning accepted by this first version.
/// These must be conservative bounds intended to contain relevant values, not just the
/// minimum and maximum observed in a small sample. The evaluator cannot verify that claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricIntervalSemanticsV1 {
    /// Caller-declared conservative envelope, interpreted as a box of independent intervals.
    DeclaredConservativeEnvelope,
}

/// Closed interval for a metric. Bounds are values, not probability quantiles.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricIntervalV1 {
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
}

/// Metric interval tied to an explicit objective dimension.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricIntervalEntryV1 {
    /// Objective dimension, reusing the canonical point-estimate metric direction.
    pub metric: ArchitectureMetricV1,
    /// Conservative interval for this dimension.
    pub interval: MetricIntervalV1,
}

/// One architecture candidate represented by one interval per required objective dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustArchitectureCandidateV1 {
    /// Stable unique identifier within this comparison.
    pub candidate_id: String,
    /// Architecture family descriptor carried by the point-estimate comparator.
    pub family: ArchitectureFamilyV1,
    /// Whether the declared architecture uses a physical tether for the task.
    pub tethered: bool,
    /// Metric entries. Every objective must appear exactly once; order is normalized in output.
    pub metrics: Vec<MetricIntervalEntryV1>,
}

/// Versioned request for robust architecture comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureRobustnessRequestV1 {
    /// Must equal ARCHITECTURE_ROBUSTNESS_SCHEMA_VERSION_V1.
    pub schema_version: u16,
    /// Stable comparison identity.
    pub comparison_id: String,
    /// Benchmark case shared by all candidates.
    pub benchmark_case_id: String,
    /// Task identity shared by all candidates.
    pub task_id: String,
    /// Wearer/device profile shared by all candidates.
    pub wearer_profile_id: String,
    /// Environment profile shared by all candidates.
    pub environment_profile_id: String,
    /// Protocol revision shared by every candidate.
    pub task_protocol_version: String,
    /// Operating conditions shared by every candidate.
    pub operating_conditions: ArchitectureOperatingConditionsV1,
    /// Caller-declared evidence level. Evaluation never promotes it.
    pub evidence_level: ArchitectureEvidenceLevelV1,
    /// Must currently be `declared_conservative_envelope`.
    pub interval_semantics: MetricIntervalSemanticsV1,
    /// Candidates under comparison, preserved in caller order.
    pub candidates: Vec<RobustArchitectureCandidateV1>,
}

/// Input validation failure. Any finding rejects the comparison instead of yielding a frontier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureRobustnessFindingV1 {
    /// Request schema is unsupported.
    UnsupportedSchemaVersion { received: u16 },
    /// Comparison identity or protocol revision is empty or surrounded by whitespace.
    InvalidIdentity { field: String },
    /// An operating-condition value is not finite.
    NonFiniteOperatingCondition { field: ArchitectureConditionFieldV1 },
    /// A non-negative operating condition is below zero.
    NegativeOperatingCondition { field: ArchitectureConditionFieldV1 },
    /// At least two candidates are required for comparison.
    InsufficientCandidates,
    /// Candidate identifier is empty or surrounded by whitespace.
    InvalidCandidateId { candidate_id: String },
    /// Candidate identifiers must be unique.
    DuplicateCandidateId { candidate_id: String },
    /// Remote-actuation architecture must declare a physical tether.
    RemoteActuationRequiresTether { candidate_id: String },
    /// An untethered architecture cannot report positive tether-interference bounds.
    TetherMetricInconsistent { candidate_id: String },
    /// A required objective was omitted.
    MissingMetric {
        candidate_id: String,
        metric: ArchitectureMetricV1,
    },
    /// A metric was supplied more than once for one candidate.
    DuplicateMetric {
        candidate_id: String,
        metric: ArchitectureMetricV1,
    },
    /// Either interval bound is not finite.
    NonFiniteBound {
        candidate_id: String,
        metric: ArchitectureMetricV1,
    },
    /// The lower bound exceeds the upper bound.
    InvertedInterval {
        candidate_id: String,
        metric: ArchitectureMetricV1,
        lower: f64,
        upper: f64,
    },
    /// The metric's domain cannot include a negative value.
    NegativeMetricBound {
        candidate_id: String,
        metric: ArchitectureMetricV1,
    },
    /// A normalized metric interval must be entirely inside [0, 1].
    OutsideNormalizedRange {
        candidate_id: String,
        metric: ArchitectureMetricV1,
        lower: f64,
        upper: f64,
    },
    /// Segment-mass envelope cannot overlap the wearer-borne-mass envelope.
    MassEnvelopeMismatch {
        candidate_id: String,
        segment_sum_lower_kg: f64,
        segment_sum_upper_kg: f64,
        wearer_mass_lower_kg: f64,
        wearer_mass_upper_kg: f64,
    },
    /// Finite mass bounds overflowed while aggregating the envelope.
    MassEnvelopeAggregationOverflow { candidate_id: String },
    /// The lower bound on total-system mass understates wearer plus off-board mass.
    TotalMassEnvelopeInconsistent {
        candidate_id: String,
        total_mass_lower_kg: f64,
        minimum_consistent_mass_kg: f64,
    },
}

/// Relation between a pair of candidates under the declared independent intervals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustPairwiseAssessmentV1 {
    /// Earlier candidate in caller input order.
    pub candidate_a_id: String,
    /// Later candidate in caller input order.
    pub candidate_b_id: String,
    /// True only if A is no worse than B across the full declared envelopes on every metric
    /// and is strictly better on at least one metric.
    pub a_robustly_dominates_b: bool,
    /// True only if B is no worse than A across the full declared envelopes on every metric
    /// and is strictly better on at least one metric.
    pub b_robustly_dominates_a: bool,
    /// Whether some assignments inside the independent intervals could make A dominate B.
    /// This is a logical possibility under the envelope model, not a probability.
    pub a_may_dominate_b_within_bounds: bool,
    /// Whether some assignments inside the independent intervals could make B dominate A.
    /// This is a logical possibility under the envelope model, not a probability.
    pub b_may_dominate_a_within_bounds: bool,
    /// Dimensions where A is strictly better than B for every value in both intervals.
    pub a_definitely_better_dimensions: Vec<ArchitectureMetricV1>,
    /// Dimensions where B is strictly better than A for every value in both intervals.
    pub b_definitely_better_dimensions: Vec<ArchitectureMetricV1>,
    /// Dimensions where neither candidate is definitely better throughout its interval.
    /// These are possible measurement targets, not guaranteed decision-changing experiments.
    pub not_separated_dimensions: Vec<ArchitectureMetricV1>,
    /// True when at least one dimension definitively favors A and another favors B.
    /// This is a robustly established trade-off, not merely overlap or missing evidence.
    pub definite_tradeoff: bool,
    /// True when neither direction robustly dominates, there is no definite trade-off,
    /// and at least one dominance direction remains logically possible within the bounds.
    pub additional_evidence_may_change_dominance: bool,
    /// Neither candidate robustly dominates the other; this alone does not imply uncertainty.
    pub unresolved: bool,
}

/// Per-candidate result on the conservative robust frontier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustCandidateAssessmentV1 {
    /// Candidate identifier from the request.
    pub candidate_id: String,
    /// Architecture family descriptor.
    pub family: ArchitectureFamilyV1,
    /// Whether the architecture declares a physical tether.
    pub tethered: bool,
    /// True if no candidate robustly dominates this candidate.
    /// Unresolved candidates remain on this conservative frontier.
    pub on_robust_frontier: bool,
    /// Candidate IDs that robustly dominate this candidate, in input order.
    pub robustly_dominated_by: Vec<String>,
    /// Candidate IDs whose comparison to this candidate remains unresolved.
    pub unresolved_against: Vec<String>,
    /// Metric intervals echoed in canonical metric order.
    pub metrics: Vec<MetricIntervalEntryV1>,
}

/// Report of robust dominance and unresolved comparisons. No scalar score is generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureRobustnessReportV1 {
    /// Report schema version.
    pub schema_version: u16,
    /// Comparison identity.
    pub comparison_id: String,
    /// Benchmark case shared by all candidates.
    pub benchmark_case_id: String,
    /// Task identity shared by all candidates.
    pub task_id: String,
    /// Wearer/device profile shared by all candidates.
    pub wearer_profile_id: String,
    /// Environment profile shared by all candidates.
    pub environment_profile_id: String,
    /// Shared task protocol revision.
    pub task_protocol_version: String,
    /// Shared operating conditions echoed from the request.
    pub operating_conditions: ArchitectureOperatingConditionsV1,
    /// Caller-declared interval interpretation.
    pub interval_semantics: MetricIntervalSemanticsV1,
    /// Evidence level copied from the request without promotion.
    pub evidence_level: ArchitectureEvidenceLevelV1,
    /// Candidate results preserve request order.
    pub candidates: Vec<RobustCandidateAssessmentV1>,
    /// One result for every unordered candidate pair, ordered deterministically.
    pub pairwise_assessments: Vec<RobustPairwiseAssessmentV1>,
    /// Explicit assumptions and claim limits.
    pub assumptions: Vec<String>,
    /// What this report can establish; no physical or clinical claim is implied.
    pub claim_ceiling: String,
}

/// Evaluate conservative metric envelopes without converting uncertainty into probability.
///
/// A robust dominator must be no worse across the full interval envelope for every required
/// metric and strictly better for at least one. Interval overlap creates an unresolved
/// comparison when neither direction can be established robustly. "May dominate" is a
/// possibility under a Cartesian-product assumption of independent intervals—not a forecast.
///
/// Required fields are validated strictly. Mass-envelope checks reject obvious contradictions,
/// but interval overlap is only a necessary consistency screen and does not recover correlations.
pub fn evaluate_architecture_robustness(
    request: &ArchitectureRobustnessRequestV1,
) -> Result<ArchitectureRobustnessReportV1, Vec<ArchitectureRobustnessFindingV1>> {
    let mut findings = Vec::new();

    if request.schema_version != ARCHITECTURE_ROBUSTNESS_SCHEMA_VERSION_V1 {
        findings.push(ArchitectureRobustnessFindingV1::UnsupportedSchemaVersion {
            received: request.schema_version,
        });
    }
    for (field, value) in [
        ("comparison_id", request.comparison_id.as_str()),
        ("benchmark_case_id", request.benchmark_case_id.as_str()),
        ("task_id", request.task_id.as_str()),
        ("wearer_profile_id", request.wearer_profile_id.as_str()),
        ("environment_profile_id", request.environment_profile_id.as_str()),
        ("task_protocol_version", request.task_protocol_version.as_str()),
    ] {
        if value.trim().is_empty() || value.trim() != value {
            findings.push(ArchitectureRobustnessFindingV1::InvalidIdentity {
                field: field.to_owned(),
            });
        }
    }

    let conditions = &request.operating_conditions;
    for (field, value) in [
        (ArchitectureConditionFieldV1::SpeedMps, conditions.speed_mps),
        (ArchitectureConditionFieldV1::PayloadKg, conditions.payload_kg),
        (ArchitectureConditionFieldV1::GradePercent, conditions.grade_percent),
        (
            ArchitectureConditionFieldV1::AmbientTemperatureC,
            conditions.ambient_temperature_c,
        ),
    ] {
        if !value.is_finite() {
            findings.push(
                ArchitectureRobustnessFindingV1::NonFiniteOperatingCondition { field },
            );
        } else if matches!(
            field,
            ArchitectureConditionFieldV1::SpeedMps | ArchitectureConditionFieldV1::PayloadKg
        ) && value < 0.0
        {
            findings.push(
                ArchitectureRobustnessFindingV1::NegativeOperatingCondition { field },
            );
        }
    }

    if request.candidates.len() < 2 {
        findings.push(ArchitectureRobustnessFindingV1::InsufficientCandidates);
    }

    let mut seen_ids = HashSet::with_capacity(request.candidates.len());
    for candidate in &request.candidates {
        if candidate.candidate_id.trim().is_empty()
            || candidate.candidate_id.trim() != candidate.candidate_id
        {
            findings.push(ArchitectureRobustnessFindingV1::InvalidCandidateId {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if !seen_ids.insert(candidate.candidate_id.as_str()) {
            findings.push(ArchitectureRobustnessFindingV1::DuplicateCandidateId {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if candidate.family == ArchitectureFamilyV1::RemoteActuationTethered
            && !candidate.tethered
        {
            findings.push(
                ArchitectureRobustnessFindingV1::RemoteActuationRequiresTether {
                    candidate_id: candidate.candidate_id.clone(),
                },
            );
        }

        let mut seen_metrics = HashSet::with_capacity(candidate.metrics.len());
        for entry in &candidate.metrics {
            if !seen_metrics.insert(entry.metric) {
                findings.push(ArchitectureRobustnessFindingV1::DuplicateMetric {
                    candidate_id: candidate.candidate_id.clone(),
                    metric: entry.metric,
                });
            }
            let lower = entry.interval.lower;
            let upper = entry.interval.upper;
            if !lower.is_finite() || !upper.is_finite() {
                findings.push(ArchitectureRobustnessFindingV1::NonFiniteBound {
                    candidate_id: candidate.candidate_id.clone(),
                    metric: entry.metric,
                });
                continue;
            }
            if lower > upper {
                findings.push(ArchitectureRobustnessFindingV1::InvertedInterval {
                    candidate_id: candidate.candidate_id.clone(),
                    metric: entry.metric,
                    lower,
                    upper,
                });
                continue;
            }
            if lower < 0.0 || upper < 0.0 {
                findings.push(ArchitectureRobustnessFindingV1::NegativeMetricBound {
                    candidate_id: candidate.candidate_id.clone(),
                    metric: entry.metric,
                });
            }
            if is_normalized_metric(entry.metric) && (lower < 0.0 || upper > 1.0) {
                findings.push(ArchitectureRobustnessFindingV1::OutsideNormalizedRange {
                    candidate_id: candidate.candidate_id.clone(),
                    metric: entry.metric,
                    lower,
                    upper,
                });
            }
        }
        for metric in REQUIRED_METRICS {
            if !seen_metrics.contains(&metric) {
                findings.push(ArchitectureRobustnessFindingV1::MissingMetric {
                    candidate_id: candidate.candidate_id.clone(),
                    metric,
                });
            }
        }
        if !candidate.tethered
            && candidate.metrics.iter().any(|entry| {
                entry.metric == ArchitectureMetricV1::TetherInterferenceEventsPerTask
                    && entry.interval.upper > 0.0
            })
        {
            findings.push(ArchitectureRobustnessFindingV1::TetherMetricInconsistent {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
    }

    // Cross-metric envelope checks run only if their needed intervals are structurally complete
    // and finite; invalid requests will be rejected below regardless.
    for candidate in &request.candidates {
        if has_valid_interval(candidate, ArchitectureMetricV1::WearerBorneMassKg)
            && has_valid_interval(candidate, ArchitectureMetricV1::TotalSystemMassKg)
            && has_valid_interval(candidate, ArchitectureMetricV1::OffboardMassKg)
            && [
                ArchitectureMetricV1::TorsoMassKg,
                ArchitectureMetricV1::PelvisMassKg,
                ArchitectureMetricV1::ProximalLimbMassKg,
                ArchitectureMetricV1::DistalLimbMassKg,
                ArchitectureMetricV1::OtherWornMassKg,
            ]
            .iter()
            .all(|metric| has_valid_interval(candidate, *metric))
        {
            validate_mass_envelopes(candidate, &mut findings);
        }
    }

    if !findings.is_empty() {
        return Err(findings);
    }

    let canonical_candidates: Vec<RobustArchitectureCandidateV1> = request
        .candidates
        .iter()
        .map(|candidate| {
            let metrics = REQUIRED_METRICS
                .iter()
                .map(|metric| MetricIntervalEntryV1 {
                    metric: *metric,
                    interval: interval_for(candidate, *metric),
                })
                .collect();
            RobustArchitectureCandidateV1 {
                candidate_id: candidate.candidate_id.clone(),
                family: candidate.family,
                tethered: candidate.tethered,
                metrics,
            }
        })
        .collect();

    let mut pairwise_assessments = Vec::new();
    let mut robustly_dominated_by: Vec<Vec<String>> =
        vec![Vec::new(); canonical_candidates.len()];
    let mut unresolved_against: Vec<Vec<String>> =
        vec![Vec::new(); canonical_candidates.len()];

    for a_index in 0..canonical_candidates.len() {
        for b_index in (a_index + 1)..canonical_candidates.len() {
            let a = &canonical_candidates[a_index];
            let b = &canonical_candidates[b_index];
            let a_robust = robustly_dominates(a, b);
            let b_robust = robustly_dominates(b, a);
            let a_may = may_dominate(a, b);
            let b_may = may_dominate(b, a);
            let unresolved = !a_robust && !b_robust;

            if a_robust {
                robustly_dominated_by[b_index].push(a.candidate_id.clone());
            }
            if b_robust {
                robustly_dominated_by[a_index].push(b.candidate_id.clone());
            }
            if unresolved {
                unresolved_against[a_index].push(b.candidate_id.clone());
                unresolved_against[b_index].push(a.candidate_id.clone());
            }

            let a_definitely_better = definitely_better_dimensions(a, b);
            let b_definitely_better = definitely_better_dimensions(b, a);
            let definite_tradeoff =
                !a_definitely_better.is_empty() && !b_definitely_better.is_empty();
            let additional_evidence_may_change_dominance = unresolved
                && !definite_tradeoff
                && (a_may || b_may);

            pairwise_assessments.push(RobustPairwiseAssessmentV1 {
                candidate_a_id: a.candidate_id.clone(),
                candidate_b_id: b.candidate_id.clone(),
                a_robustly_dominates_b: a_robust,
                b_robustly_dominates_a: b_robust,
                a_may_dominate_b_within_bounds: a_may,
                b_may_dominate_a_within_bounds: b_may,
                not_separated_dimensions: not_separated_dimensions(a, b),
                a_definitely_better_dimensions: a_definitely_better,
                b_definitely_better_dimensions: b_definitely_better,
                definite_tradeoff,
                additional_evidence_may_change_dominance,
                unresolved,
            });

        }
    }

    let candidates = canonical_candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| RobustCandidateAssessmentV1 {
            candidate_id: candidate.candidate_id,
            family: candidate.family,
            tethered: candidate.tethered,
            on_robust_frontier: robustly_dominated_by[index].is_empty(),
            robustly_dominated_by: robustly_dominated_by[index].clone(),
            unresolved_against: unresolved_against[index].clone(),
            metrics: candidate.metrics,
        })
        .collect();

    Ok(ArchitectureRobustnessReportV1 {
        schema_version: ARCHITECTURE_ROBUSTNESS_SCHEMA_VERSION_V1,
        comparison_id: request.comparison_id.clone(),
        benchmark_case_id: request.benchmark_case_id.clone(),
        task_id: request.task_id.clone(),
        wearer_profile_id: request.wearer_profile_id.clone(),
        environment_profile_id: request.environment_profile_id.clone(),
        task_protocol_version: request.task_protocol_version.clone(),
        operating_conditions: request.operating_conditions.clone(),
        interval_semantics: request.interval_semantics,
        evidence_level: request.evidence_level,
        candidates,
        pairwise_assessments,
        assumptions: vec![
            "Bounds are caller-declared conservative envelopes and are not authenticated."
                .to_owned(),
            "Intervals are treated as independent; cross-metric and within-person"
                .to_owned()
                + " correlations are not modeled.",
            "May-dominate results are logical possibilities, not probabilities or predictions."
                .to_owned(),
            "Observed sample ranges and confidence intervals are not automatically valid"
                .to_owned()
                + " conservative envelopes.",
            "Mass-envelope consistency checks are necessary screens and do not prove"
                .to_owned()
                + " physical consistency.",
            "This is report-only research software; it emits no actuator commands"
                .to_owned()
                + " and is not a safety case.",
        ],
        claim_ceiling: concat!(
            "Interval-aware synthetic/replay screening only. No physical, clinical, ",
            "safety, regulatory, or product-performance claim."
        )
        .to_owned(),
    })
}

fn is_higher_better(metric: ArchitectureMetricV1) -> bool {
    matches!(
        metric,
        ArchitectureMetricV1::TaskSuccessRate | ArchitectureMetricV1::ComfortScore
    )
}

fn is_normalized_metric(metric: ArchitectureMetricV1) -> bool {
    matches!(
        metric,
        ArchitectureMetricV1::TaskSuccessRate
            | ArchitectureMetricV1::MovementRestrictionIndex
            | ArchitectureMetricV1::ComfortScore
    )
}

fn has_valid_interval(
    candidate: &RobustArchitectureCandidateV1,
    metric: ArchitectureMetricV1,
) -> bool {
    let matching: Vec<&MetricIntervalV1> = candidate
        .metrics
        .iter()
        .filter(|entry| entry.metric == metric)
        .map(|entry| &entry.interval)
        .collect();
    matching.len() == 1
        && matching[0].lower.is_finite()
        && matching[0].upper.is_finite()
        && matching[0].lower <= matching[0].upper
}

fn interval_for(
    candidate: &RobustArchitectureCandidateV1,
    metric: ArchitectureMetricV1,
) -> MetricIntervalV1 {
    candidate
        .metrics
        .iter()
        .find(|entry| entry.metric == metric)
        .expect("candidate intervals are validated before evaluation")
        .interval
}

fn validate_mass_envelopes(
    candidate: &RobustArchitectureCandidateV1,
    findings: &mut Vec<ArchitectureRobustnessFindingV1>,
) {
    let segment_metrics = [
        ArchitectureMetricV1::TorsoMassKg,
        ArchitectureMetricV1::PelvisMassKg,
        ArchitectureMetricV1::ProximalLimbMassKg,
        ArchitectureMetricV1::DistalLimbMassKg,
        ArchitectureMetricV1::OtherWornMassKg,
    ];
    let mut segment_lower = 0.0_f64;
    let mut segment_upper = 0.0_f64;
    for metric in segment_metrics {
        let interval = interval_for(candidate, metric);
        segment_lower += interval.lower;
        segment_upper += interval.upper;
    }

    let wearer = interval_for(candidate, ArchitectureMetricV1::WearerBorneMassKg);
    if !segment_lower.is_finite() || !segment_upper.is_finite() {
        findings.push(ArchitectureRobustnessFindingV1::MassEnvelopeAggregationOverflow {
            candidate_id: candidate.candidate_id.clone(),
        });
        return;
    }
    if segment_upper < wearer.lower - ROBUST_MASS_BALANCE_TOLERANCE_KG_V1
        || segment_lower > wearer.upper + ROBUST_MASS_BALANCE_TOLERANCE_KG_V1
    {
        findings.push(ArchitectureRobustnessFindingV1::MassEnvelopeMismatch {
            candidate_id: candidate.candidate_id.clone(),
            segment_sum_lower_kg: segment_lower,
            segment_sum_upper_kg: segment_upper,
            wearer_mass_lower_kg: wearer.lower,
            wearer_mass_upper_kg: wearer.upper,
        });
    }

    let offboard = interval_for(candidate, ArchitectureMetricV1::OffboardMassKg);
    let total = interval_for(candidate, ArchitectureMetricV1::TotalSystemMassKg);
    let minimum_consistent_mass = wearer.lower + offboard.lower;
    if !minimum_consistent_mass.is_finite() {
        findings.push(ArchitectureRobustnessFindingV1::MassEnvelopeAggregationOverflow {
            candidate_id: candidate.candidate_id.clone(),
        });
    } else if total.lower + ROBUST_MASS_BALANCE_TOLERANCE_KG_V1 < minimum_consistent_mass {
        findings.push(ArchitectureRobustnessFindingV1::TotalMassEnvelopeInconsistent {
            candidate_id: candidate.candidate_id.clone(),
            total_mass_lower_kg: total.lower,
            minimum_consistent_mass_kg: minimum_consistent_mass,
        });
    }
}

fn robustly_dominates(
    a: &RobustArchitectureCandidateV1,
    b: &RobustArchitectureCandidateV1,
) -> bool {
    let mut no_worse_on_all = true;
    let mut strict_on_at_least_one = false;
    for metric in REQUIRED_METRICS {
        let a_interval = interval_for(a, metric);
        let b_interval = interval_for(b, metric);
        let (no_worse, strictly_better) = if is_higher_better(metric) {
            (
                a_interval.lower >= b_interval.upper - ROBUST_COMPARISON_EPSILON_V1,
                a_interval.lower > b_interval.upper + ROBUST_COMPARISON_EPSILON_V1,
            )
        } else {
            (
                a_interval.upper <= b_interval.lower + ROBUST_COMPARISON_EPSILON_V1,
                a_interval.upper < b_interval.lower - ROBUST_COMPARISON_EPSILON_V1,
            )
        };
        no_worse_on_all &= no_worse;
        strict_on_at_least_one |= strictly_better;
    }
    no_worse_on_all && strict_on_at_least_one
}

fn may_dominate(
    a: &RobustArchitectureCandidateV1,
    b: &RobustArchitectureCandidateV1,
) -> bool {
    let mut no_worse_possible_on_all = true;
    let mut strict_possible = false;
    for metric in REQUIRED_METRICS {
        let a_interval = interval_for(a, metric);
        let b_interval = interval_for(b, metric);
        let (no_worse_possible, strict_possible_here) = if is_higher_better(metric) {
            (
                a_interval.upper >= b_interval.lower - ROBUST_COMPARISON_EPSILON_V1,
                a_interval.upper > b_interval.lower + ROBUST_COMPARISON_EPSILON_V1,
            )
        } else {
            (
                a_interval.lower <= b_interval.upper + ROBUST_COMPARISON_EPSILON_V1,
                a_interval.lower < b_interval.upper - ROBUST_COMPARISON_EPSILON_V1,
            )
        };
        no_worse_possible_on_all &= no_worse_possible;
        strict_possible |= strict_possible_here;
    }
    no_worse_possible_on_all && strict_possible
}

fn not_separated_dimensions(
    a: &RobustArchitectureCandidateV1,
    b: &RobustArchitectureCandidateV1,
) -> Vec<ArchitectureMetricV1> {
    REQUIRED_METRICS
        .into_iter()
        .filter(|metric| {
            let a_interval = interval_for(a, *metric);
            let b_interval = interval_for(b, *metric);
            let a_better = if is_higher_better(*metric) {
                a_interval.lower > b_interval.upper + ROBUST_COMPARISON_EPSILON_V1
            } else {
                a_interval.upper < b_interval.lower - ROBUST_COMPARISON_EPSILON_V1
            };
            let b_better = if is_higher_better(*metric) {
                b_interval.lower > a_interval.upper + ROBUST_COMPARISON_EPSILON_V1
            } else {
                b_interval.upper < a_interval.lower - ROBUST_COMPARISON_EPSILON_V1
            };
            !a_better && !b_better
        })
        .collect()
}

fn definitely_better_dimensions(
    a: &RobustArchitectureCandidateV1,
    b: &RobustArchitectureCandidateV1,
) -> Vec<ArchitectureMetricV1> {
    REQUIRED_METRICS
        .into_iter()
        .filter(|metric| {
            let a_interval = interval_for(a, *metric);
            let b_interval = interval_for(b, *metric);
            if is_higher_better(*metric) {
                a_interval.lower > b_interval.upper + ROBUST_COMPARISON_EPSILON_V1
            } else {
                a_interval.upper < b_interval.lower - ROBUST_COMPARISON_EPSILON_V1
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    const ROBUSTNESS_FIXTURE_JSON: &str = include_str!(
        "../../../../docs/research/open-mobility/fixtures/architecture-robustness-scenarios-v1.json"
    );

    #[derive(Clone, Copy)]
    enum Profile {
        Balanced,
        Overbuilt,
        MassFirst,
    }

    fn metric_interval(metric: ArchitectureMetricV1, profile: Profile) -> MetricIntervalV1 {
        use ArchitectureMetricV1 as M;
        match (metric, profile) {
            (M::TaskSuccessRate, Profile::Balanced) => MetricIntervalV1 {
                lower: 0.90,
                upper: 0.95,
            },
            (M::TaskSuccessRate, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 0.65,
                upper: 0.75,
            },
            (M::TaskSuccessRate, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.55,
                upper: 0.62,
            },
            (M::ComfortScore, Profile::Balanced) => MetricIntervalV1 {
                lower: 0.82,
                upper: 0.88,
            },
            (M::ComfortScore, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 0.50,
                upper: 0.60,
            },
            (M::ComfortScore, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.55,
                upper: 0.65,
            },
            (M::MovementRestrictionIndex, Profile::Balanced) => MetricIntervalV1 {
                lower: 0.10,
                upper: 0.20,
            },
            (M::MovementRestrictionIndex, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 0.45,
                upper: 0.55,
            },
            (M::MovementRestrictionIndex, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.05,
                upper: 0.10,
            },
            (M::WearerBorneMassKg, Profile::Balanced) => MetricIntervalV1 {
                lower: 4.5,
                upper: 5.0,
            },
            (M::WearerBorneMassKg, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 7.7,
                upper: 8.2,
            },
            (M::WearerBorneMassKg, Profile::MassFirst) => MetricIntervalV1 {
                lower: 2.0,
                upper: 2.4,
            },
            (M::TotalSystemMassKg, Profile::Balanced) => MetricIntervalV1 {
                lower: 5.2,
                upper: 5.7,
            },
            (M::TotalSystemMassKg, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 8.8,
                upper: 9.3,
            },
            (M::TotalSystemMassKg, Profile::MassFirst) => MetricIntervalV1 {
                lower: 2.3,
                upper: 2.8,
            },
            (M::OffboardMassKg, Profile::Balanced) => MetricIntervalV1 {
                lower: 0.3,
                upper: 0.5,
            },
            (M::OffboardMassKg, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 0.5,
                upper: 0.7,
            },
            (M::OffboardMassKg, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.1,
                upper: 0.2,
            },
            (M::TorsoMassKg, Profile::Balanced) => MetricIntervalV1 {
                lower: 0.5,
                upper: 0.7,
            },
            (M::TorsoMassKg, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 0.9,
                upper: 1.1,
            },
            (M::TorsoMassKg, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.2,
                upper: 0.3,
            },
            (M::PelvisMassKg, Profile::Balanced) => MetricIntervalV1 {
                lower: 0.8,
                upper: 1.0,
            },
            (M::PelvisMassKg, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 1.4,
                upper: 1.6,
            },
            (M::PelvisMassKg, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.3,
                upper: 0.5,
            },
            (M::ProximalLimbMassKg, Profile::Balanced) => MetricIntervalV1 {
                lower: 1.1,
                upper: 1.3,
            },
            (M::ProximalLimbMassKg, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 2.0,
                upper: 2.2,
            },
            (M::ProximalLimbMassKg, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.5,
                upper: 0.7,
            },
            (M::DistalLimbMassKg, Profile::Balanced) => MetricIntervalV1 {
                lower: 1.0,
                upper: 1.2,
            },
            (M::DistalLimbMassKg, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 2.0,
                upper: 2.3,
            },
            (M::DistalLimbMassKg, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.5,
                upper: 0.7,
            },
            (M::OtherWornMassKg, Profile::Balanced) => MetricIntervalV1 {
                lower: 0.4,
                upper: 0.5,
            },
            (M::OtherWornMassKg, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 0.6,
                upper: 0.8,
            },
            (M::OtherWornMassKg, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.3,
                upper: 0.4,
            },
            (M::EnergyWhPerTask, Profile::Balanced) => MetricIntervalV1 {
                lower: 1.0,
                upper: 1.2,
            },
            (M::EnergyWhPerTask, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 2.0,
                upper: 2.2,
            },
            (M::EnergyWhPerTask, Profile::MassFirst) => MetricIntervalV1 {
                lower: 0.4,
                upper: 0.6,
            },
            (M::SetupTimeSeconds, Profile::Balanced) => MetricIntervalV1 {
                lower: 40.0,
                upper: 45.0,
            },
            (M::SetupTimeSeconds, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 70.0,
                upper: 80.0,
            },
            (M::SetupTimeSeconds, Profile::MassFirst) => MetricIntervalV1 {
                lower: 20.0,
                upper: 30.0,
            },
            (M::TetherInterferenceEventsPerTask, Profile::Balanced) => {
                MetricIntervalV1 { lower: 0.0, upper: 0.0 }
            }
            (M::TetherInterferenceEventsPerTask, Profile::Overbuilt) => {
                MetricIntervalV1 { lower: 0.0, upper: 0.0 }
            }
            (M::TetherInterferenceEventsPerTask, Profile::MassFirst) => {
                MetricIntervalV1 { lower: 0.0, upper: 0.0 }
            },
            (M::RepairTimeMinutes, Profile::Balanced) => MetricIntervalV1 {
                lower: 20.0,
                upper: 25.0,
            },
            (M::RepairTimeMinutes, Profile::Overbuilt) => MetricIntervalV1 {
                lower: 40.0,
                upper: 50.0,
            },
            (M::RepairTimeMinutes, Profile::MassFirst) => MetricIntervalV1 {
                lower: 10.0,
                upper: 15.0,
            },
        }
    }

    fn candidate(candidate_id: &str, profile: Profile) -> RobustArchitectureCandidateV1 {
        RobustArchitectureCandidateV1 {
            candidate_id: candidate_id.to_owned(),
            family: match profile {
                Profile::Balanced => ArchitectureFamilyV1::HybridPoweredPassive,
                Profile::Overbuilt => ArchitectureFamilyV1::PoweredRigid,
                Profile::MassFirst => ArchitectureFamilyV1::Passive,
            },
            tethered: false,
            metrics: REQUIRED_METRICS
                .into_iter()
                .map(|metric| MetricIntervalEntryV1 {
                    metric,
                    interval: metric_interval(metric, profile),
                })
                .collect(),
        }
    }

    fn request(candidates: Vec<RobustArchitectureCandidateV1>) -> ArchitectureRobustnessRequestV1 {
        ArchitectureRobustnessRequestV1 {
            schema_version: ARCHITECTURE_ROBUSTNESS_SCHEMA_VERSION_V1,
            comparison_id: "robust-fixture-v1".to_owned(),
            benchmark_case_id: "walk-case-profile-a".to_owned(),
            task_id: "overground_walk".to_owned(),
            wearer_profile_id: "profile-a".to_owned(),
            environment_profile_id: "flat_lab".to_owned(),
            task_protocol_version: "walk-protocol-v1".to_owned(),
            operating_conditions: ArchitectureOperatingConditionsV1 {
                speed_mps: 0.8,
                payload_kg: 0.0,
                grade_percent: 0.0,
                ambient_temperature_c: 22.0,
            },
            evidence_level: ArchitectureEvidenceLevelV1::SyntheticReplay,
            interval_semantics: MetricIntervalSemanticsV1::DeclaredConservativeEnvelope,
            candidates,
        }
    }

    #[test]
    fn clearly_separated_envelopes_yield_robust_dominance_and_explanation() {
        let report = evaluate_architecture_robustness(&request(vec![
            candidate("balanced", Profile::Balanced),
            candidate("overbuilt", Profile::Overbuilt),
        ]))
        .expect("well-formed synthetic envelopes");

        assert!(report.pairwise_assessments[0].a_robustly_dominates_b);
        assert!(!report.pairwise_assessments[0].b_robustly_dominates_a);
        assert!(!report.pairwise_assessments[0].unresolved);
        assert!(report.pairwise_assessments[0].a_definitely_better_dimensions.len() >= 10);
        assert!(report.candidates[0].on_robust_frontier);
        assert!(!report.candidates[1].on_robust_frontier);
        assert_eq!(
            report.candidates[1].robustly_dominated_by,
            vec!["balanced".to_owned()]
        );
    }

    #[test]
    fn mass_first_architecture_remains_frontier_due_to_real_tradeoffs() {
        let report = evaluate_architecture_robustness(&request(vec![
            candidate("balanced", Profile::Balanced),
            candidate("mass-first", Profile::MassFirst),
            candidate("overbuilt", Profile::Overbuilt),
        ]))
        .expect("well-formed synthetic envelopes");

        assert!(report.candidates[0].on_robust_frontier);
        assert!(report.candidates[1].on_robust_frontier);
        assert!(!report.candidates[2].on_robust_frontier);
        assert_eq!(
            report
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == "overbuilt")
                .expect("overbuilt assessment exists")
                .robustly_dominated_by,
            vec!["balanced".to_owned()]
        );
        let tradeoff = report
            .pairwise_assessments
            .iter()
            .find(|pair| {
                pair.candidate_a_id == "balanced"
                    && pair.candidate_b_id == "mass-first"
            })
            .expect("balanced/mass-first pair exists");
        assert!(tradeoff.unresolved);
        assert!(!tradeoff.a_robustly_dominates_b && !tradeoff.b_robustly_dominates_a);
        assert!(tradeoff.definite_tradeoff);
        assert!(!tradeoff.additional_evidence_may_change_dominance);
        assert!(!tradeoff.a_definitely_better_dimensions.is_empty());
        assert!(!tradeoff.b_definitely_better_dimensions.is_empty());
    }

    #[test]
    fn overlap_never_becomes_a_robust_win_and_may_dominate_is_not_probability() {
        let report = evaluate_architecture_robustness(&request(vec![
            candidate("overlap-a", Profile::Balanced),
            candidate("overlap-b", Profile::Balanced),
        ]))
        .expect("identical interval envelopes are valid");

        let pair = &report.pairwise_assessments[0];
        assert!(!pair.a_robustly_dominates_b);
        assert!(!pair.b_robustly_dominates_a);
        assert!(pair.a_may_dominate_b_within_bounds);
        assert!(pair.b_may_dominate_a_within_bounds);
        assert!(pair.unresolved);
        assert!(!pair.definite_tradeoff);
        assert!(pair.additional_evidence_may_change_dominance);
        assert_eq!(pair.not_separated_dimensions.len(), REQUIRED_METRICS.len());
    }

    #[test]
    fn widening_any_single_metric_envelope_cannot_preserve_false_certainty() {
        for metric in REQUIRED_METRICS {
            let mut a = candidate("candidate-a", Profile::Balanced);
            let mut b = candidate("candidate-b", Profile::Overbuilt);
            if metric == ArchitectureMetricV1::TetherInterferenceEventsPerTask {
                a.family = ArchitectureFamilyV1::RemoteActuationTethered;
                a.tethered = true;
                b.family = ArchitectureFamilyV1::RemoteActuationTethered;
                b.tethered = true;
            }
            let baseline = evaluate_architecture_robustness(&request(vec![a.clone(), b.clone()]))
                .expect("baseline envelopes are valid");
            assert!(baseline.pairwise_assessments[0].a_robustly_dominates_b);

            let b_interval = b
                .metrics
                .iter()
                .find(|entry| entry.metric == metric)
                .expect("required metric exists")
                .interval;
            let a_interval = &mut a
                .metrics
                .iter_mut()
                .find(|entry| entry.metric == metric)
                .expect("required metric exists")
                .interval;
            if is_higher_better(metric) {
                a_interval.lower = (b_interval.upper - 0.1).max(0.0);
            } else {
                a_interval.upper = b_interval.lower + 0.1;
            }

            let report = evaluate_architecture_robustness(&request(vec![a, b]))
                .expect("one-sided envelope widening preserves fixture integrity");
            let pair = &report.pairwise_assessments[0];
            assert!(
                !pair.a_robustly_dominates_b,
                "widening {metric:?} across the rival bound must erase robust dominance"
            );
        }
    }

    #[test]
    fn invalid_tether_declarations_fail_closed() {
        let mut untethered_with_interference = candidate("untethered", Profile::Balanced);
        untethered_with_interference
            .metrics
            .iter_mut()
            .find(|entry| {
                entry.metric == ArchitectureMetricV1::TetherInterferenceEventsPerTask
            })
            .expect("tether metric exists")
            .interval = MetricIntervalV1 {
            lower: 0.0,
            upper: 0.01,
        };
        let result = evaluate_architecture_robustness(&request(vec![
            untethered_with_interference,
            candidate("other", Profile::Overbuilt),
        ]));
        assert!(result.as_ref().err().unwrap().iter().any(|finding| {
            matches!(
                finding,
                ArchitectureRobustnessFindingV1::TetherMetricInconsistent { .. }
            )
        }));

        let mut remote_without_tether = candidate("remote", Profile::Balanced);
        remote_without_tether.family = ArchitectureFamilyV1::RemoteActuationTethered;
        let result = evaluate_architecture_robustness(&request(vec![
            remote_without_tether,
            candidate("other", Profile::Overbuilt),
        ]));
        assert!(result.as_ref().err().unwrap().iter().any(|finding| {
            matches!(
                finding,
                ArchitectureRobustnessFindingV1::RemoteActuationRequiresTether { .. }
            )
        }));
    }

    #[test]
    fn invalid_operating_conditions_and_identity_fail_closed() {
        let baseline = request(vec![
            candidate("balanced", Profile::Balanced),
            candidate("overbuilt", Profile::Overbuilt),
        ]);

        let mut negative_payload = baseline.clone();
        negative_payload.operating_conditions.payload_kg = -0.1;
        assert!(evaluate_architecture_robustness(&negative_payload)
            .as_ref()
            .err()
            .unwrap()
            .iter()
            .any(|finding| {
                matches!(
                    finding,
                    ArchitectureRobustnessFindingV1::NegativeOperatingCondition {
                        field: ArchitectureConditionFieldV1::PayloadKg
                    }
                )
            }));

        let mut nonfinite_speed = baseline.clone();
        nonfinite_speed.operating_conditions.speed_mps = f64::INFINITY;
        assert!(evaluate_architecture_robustness(&nonfinite_speed)
            .as_ref()
            .err()
            .unwrap()
            .iter()
            .any(|finding| {
                matches!(
                    finding,
                    ArchitectureRobustnessFindingV1::NonFiniteOperatingCondition {
                        field: ArchitectureConditionFieldV1::SpeedMps
                    }
                )
            }));

        let mut missing_identity = baseline;
        missing_identity.task_id = " overground_walk".to_owned();
        assert!(evaluate_architecture_robustness(&missing_identity)
            .as_ref()
            .err()
            .unwrap()
            .iter()
            .any(|finding| {
                matches!(
                    finding,
                    ArchitectureRobustnessFindingV1::InvalidIdentity { field }
                        if field == "task_id"
                )
            }));
    }

    #[test]
    fn missing_and_duplicate_metric_dimensions_fail_closed() {
        let mut missing = candidate("missing", Profile::Balanced);
        missing.metrics.pop();
        let missing_result = evaluate_architecture_robustness(&request(vec![
            missing,
            candidate("other", Profile::Overbuilt),
        ]));
        assert!(missing_result.as_ref().err().unwrap().iter().any(|finding| {
            matches!(finding, ArchitectureRobustnessFindingV1::MissingMetric { .. })
        }));

        let mut duplicate = candidate("duplicate", Profile::Balanced);
        duplicate.metrics.push(duplicate.metrics[0]);
        let duplicate_result = evaluate_architecture_robustness(&request(vec![
            duplicate,
            candidate("other", Profile::Overbuilt),
        ]));
        assert!(duplicate_result.as_ref().err().unwrap().iter().any(|finding| {
            matches!(finding, ArchitectureRobustnessFindingV1::DuplicateMetric { .. })
        }));
    }

    #[test]
    fn nonfinite_inverted_and_normalized_out_of_range_bounds_fail_closed() {
        let mut nonfinite = candidate("nonfinite", Profile::Balanced);
        nonfinite.metrics[0].interval.upper = f64::INFINITY;
        assert!(evaluate_architecture_robustness(&request(vec![
            nonfinite,
            candidate("other", Profile::Overbuilt),
        ]))
        .as_ref()
        .err()
        .unwrap()
        .iter()
        .any(|finding| matches!(finding, ArchitectureRobustnessFindingV1::NonFiniteBound { .. })));

        let mut inverted = candidate("inverted", Profile::Balanced);
        inverted.metrics[0].interval = MetricIntervalV1 { lower: 2.0, upper: 1.0 };
        assert!(evaluate_architecture_robustness(&request(vec![
            inverted,
            candidate("other", Profile::Overbuilt),
        ]))
        .as_ref()
        .err()
        .unwrap()
        .iter()
        .any(|finding| {
            matches!(
                finding,
                ArchitectureRobustnessFindingV1::InvertedInterval { .. }
            )
        }));

        let mut normalized = candidate("normalized", Profile::Balanced);
        normalized
            .metrics
            .iter_mut()
            .find(|entry| entry.metric == ArchitectureMetricV1::ComfortScore)
            .expect("comfort metric exists")
            .interval =
            MetricIntervalV1 { lower: 0.8, upper: 1.1 };
        assert!(evaluate_architecture_robustness(&request(vec![
            normalized,
            candidate("other", Profile::Overbuilt),
        ]))
        .as_ref()
        .err()
        .unwrap()
        .iter()
        .any(|finding| {
            matches!(
                finding,
                ArchitectureRobustnessFindingV1::OutsideNormalizedRange { .. }
            )
        }));
    }

    #[test]
    fn inconsistent_mass_envelopes_are_rejected() {
        let mut inconsistent = candidate("mass-invalid", Profile::Balanced);
        inconsistent
            .metrics
            .iter_mut()
            .find(|entry| entry.metric == ArchitectureMetricV1::WearerBorneMassKg)
            .expect("wearer mass metric exists")
            .interval =
            MetricIntervalV1 { lower: 40.0, upper: 50.0 };
        assert!(evaluate_architecture_robustness(&request(vec![
            inconsistent,
            candidate("other", Profile::Overbuilt),
        ]))
        .as_ref()
        .err()
        .unwrap()
        .iter()
        .any(|finding| {
            matches!(
                finding,
                ArchitectureRobustnessFindingV1::MassEnvelopeMismatch { .. }
            )
        }));
    }

    #[derive(Deserialize)]
    struct FixtureV1 {
        request: ArchitectureRobustnessRequestV1,
        expected_robust_frontier: Vec<String>,
        expected_dominance: Vec<ExpectedDominanceV1>,
        expected_unresolved_pairs: Vec<[String; 2]>,
    }

    #[derive(Deserialize)]
    struct ExpectedDominanceV1 {
        dominator: String,
        dominated: String,
    }

    #[test]
    fn versioned_fixture_matches_expected_frontier_and_pairwise_results() {
        let fixture: FixtureV1 =
            serde_json::from_str(ROBUSTNESS_FIXTURE_JSON).expect("fixture JSON must parse");
        let report = evaluate_architecture_robustness(&fixture.request)
            .expect("versioned synthetic fixture must validate");

        let frontier: Vec<String> = report
            .candidates
            .iter()
            .filter(|candidate| candidate.on_robust_frontier)
            .map(|candidate| candidate.candidate_id.clone())
            .collect();
        assert_eq!(frontier, fixture.expected_robust_frontier);

        for expected in fixture.expected_dominance {
            let result = report.pairwise_assessments.iter().find(|pair| {
                pair.candidate_a_id == expected.dominator
                    && pair.candidate_b_id == expected.dominated
            }).expect("expected pair must be present in canonical pair order");
            assert!(result.a_robustly_dominates_b);
        }
        for expected in fixture.expected_unresolved_pairs {
            let result = report.pairwise_assessments.iter().find(|pair| {
                pair.candidate_a_id == expected[0] && pair.candidate_b_id == expected[1]
            }).expect("expected unresolved pair must be present");
            assert!(result.unresolved);
            assert!(!result.a_robustly_dominates_b && !result.b_robustly_dominates_a);
        }
    }

    #[test]
    fn repeated_evaluation_is_deterministic_and_reports_no_probability() {
        let input = request(vec![
            candidate("balanced", Profile::Balanced),
            candidate("overbuilt", Profile::Overbuilt),
            candidate("mass-first", Profile::MassFirst),
        ]);
        let expected = evaluate_architecture_robustness(&input).expect("valid request");
        let bytes = serde_json::to_vec(&expected).expect("report serializes");
        for _ in 0..100 {
            let actual = evaluate_architecture_robustness(&input).expect("valid request");
            assert_eq!(actual, expected);
            assert_eq!(serde_json::to_vec(&actual).expect("report serializes"), bytes);
        }
        assert!(expected.assumptions.iter().any(|line| line.contains("not probabilities")));
    }
}
