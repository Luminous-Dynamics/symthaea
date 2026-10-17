// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Decision-sensitive measurement triage over interval-aware exosuit comparisons.
//!
//! This module consumes an existing robust-comparison report and a caller-declared action
//! catalog. It does not schedule, authorize, or execute experiments; estimate information gain;
//! infer likelihoods; or authenticate feasibility, consent, safety, calibration, or provenance.
//! It preserves a set of non-dominated measurement proposals instead of choosing one scalar winner.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::architecture_robustness::{
    ArchitectureRobustnessReportV1, ARCHITECTURE_ROBUSTNESS_SCHEMA_VERSION_V1,
};
use crate::architecture_pareto::{
    ArchitectureEvidenceLevelV1, ArchitectureMetricV1,
};

/// Current schema version for decision-sensitive measurement triage.
pub const MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1: u16 = 1;
/// Floating-point tolerance used only for resource-cost Pareto comparison.
pub const MEASUREMENT_TRIAGE_EPSILON_V1: f64 = 1.0e-9;

/// Type of proposed information-gathering work. This is not an authorization to execute it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementActionKindV1 {
    /// Offline analysis of already admitted evidence.
    Analysis,
    /// A deterministic simulation or replay that does not operate physical hardware.
    Simulation,
    /// A physical bench measurement requiring a current calibration chain and safety review.
    BenchMeasurement,
    /// A human-participant study requiring ethics/consent and safety prerequisites.
    HumanStudy,
}

/// Status supplied for a required action prerequisite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementGateStatusV1 {
    /// Caller asserts that the prerequisite is satisfied for this exact protocol/action.
    Satisfied,
    /// Prerequisite is explicitly not satisfied.
    Blocked,
    /// Status is unknown; triage fails closed.
    Unknown,
    /// Gate does not apply to this action kind.
    NotApplicable,
}

/// Caller-declared prerequisites. These flags are checked, not authenticated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementActionGatesV1 {
    /// The action's protocol and decision rule are frozen before results are observed.
    pub protocol_frozen: MeasurementGateStatusV1,
    /// All candidates in each claimed comparison can be measured under matched conditions.
    pub candidate_coverage: MeasurementGateStatusV1,
    /// The proposed action is feasible in the declared setup/resource envelope.
    pub feasibility_review: MeasurementGateStatusV1,
    /// Appropriate risk/safety review is satisfied for the action kind.
    pub safety_review: MeasurementGateStatusV1,
    /// Required consent/ethics authorization is satisfied for participant involvement.
    pub consent_review: MeasurementGateStatusV1,
    /// Applicable instruments have a current, relevant calibration status.
    pub calibration_status: MeasurementGateStatusV1,
    /// The evidence path and source lineage are defined for the planned output.
    pub evidence_provenance: MeasurementGateStatusV1,
}

/// Declared range for time/cost burden. It is not an expected value distribution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementCostIntervalV1 {
    /// Inclusive best-case planning estimate in minutes.
    pub lower_minutes: f64,
    /// Inclusive conservative planning estimate in minutes.
    pub upper_minutes: f64,
}

/// One proposed measurement or analysis action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementActionV1 {
    /// Stable unique action ID in this catalog.
    pub action_id: String,
    /// Declared action kind.
    pub kind: MeasurementActionKindV1,
    /// Must match the exact comparison report's task-protocol revision.
    pub task_protocol_version: String,
    /// Candidates that this action can measure under matched conditions.
    pub candidate_ids: Vec<String>,
    /// Metrics the action can measure or refine for every listed candidate.
    pub metric_targets: Vec<ArchitectureMetricV1>,
    /// Planning-cost interval in minutes.
    pub cost_minutes: MeasurementCostIntervalV1,
    /// Caller-declared prerequisite statuses.
    pub gates: MeasurementActionGatesV1,
}

/// One actionable metric/pair cell. Coverage does not imply that the comparison will resolve.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSensitiveCellV1 {
    /// Earlier candidate in the source report's pair order.
    pub candidate_a_id: String,
    /// Later candidate in the source report's pair order.
    pub candidate_b_id: String,
    /// Metric not separated by the declared intervals.
    pub metric: ArchitectureMetricV1,
}

/// Why a proposed action is not eligible for the triage frontier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementActionBlockerV1 {
    /// Prerequisite that blocks eligibility.
    pub gate: String,
    /// Caller-declared status.
    pub status: MeasurementGateStatusV1,
    /// Explanation of why this status is not eligible.
    pub reason: String,
}

/// Result for one proposal. All actions remain proposals; none are executed or authorized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementActionAssessmentV1 {
    /// Action identifier.
    pub action_id: String,
    /// Action kind.
    pub kind: MeasurementActionKindV1,
    /// True only when required gate statuses are satisfied and the action covers at least one
    /// metric-pair cell whose dominance could change within declared bounds.
    pub eligible_for_triage: bool,
    /// Gate findings or a no-decision-relevant-coverage finding.
    pub blockers: Vec<MeasurementActionBlockerV1>,
    /// Exact metric/pair cells this catalog entry claims it can measure.
    pub covered_cells: Vec<DecisionSensitiveCellV1>,
    /// Number of unique covered cells; a deterministic coverage diagnostic, not expected value.
    pub covered_cell_count: usize,
    /// True when no eligible action has a superset of its coverage at no greater upper cost.
    pub on_proposal_frontier: bool,
    /// Eligible actions that dominate this action under coverage-superset and upper-cost rules.
    pub dominated_by: Vec<String>,
    /// Planning cost echoed from the action catalog.
    pub cost_minutes: MeasurementCostIntervalV1,
}

/// Decision-sensitive triage report over one exact robust-comparison receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementTriageReportV1 {
    /// Triage report schema.
    pub schema_version: u16,
    /// Source robust-comparison identity.
    pub comparison_id: String,
    /// Exact shared task-protocol revision.
    pub task_protocol_version: String,
    /// Source evidence level; never promoted.
    pub evidence_level: ArchitectureEvidenceLevelV1,
    /// Count of pairs for which more evidence could change the dominance relation under bounds.
    pub measurement_sensitive_pair_count: usize,
    /// Count of metric-pair cells available for targeted measurement under the source report.
    pub decision_sensitive_cell_count: usize,
    /// Assessments preserve action-catalog order.
    pub actions: Vec<MeasurementActionAssessmentV1>,
    /// Explicit assumptions.
    pub assumptions: Vec<String>,
    /// Triage claim ceiling.
    pub claim_ceiling: String,
}

/// Input rejection. No ranking is produced from an invalid source report or action catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementTriageFindingV1 {
    /// Unsupported triage schema.
    UnsupportedSchemaVersion { received: u16 },
    /// Unsupported schema in the source robustness report.
    UnsupportedSourceSchemaVersion { received: u16 },
    /// Source comparison has fewer than two candidates or incomplete pair coverage.
    IncompleteSourceComparison,
    /// Source candidate or action ID is empty, padded, or duplicated.
    InvalidOrDuplicateIdentity { identity: String },
    /// A pairwise report refers to an unknown candidate or repeats an unordered pair.
    InvalidOrDuplicatePair { candidate_a_id: String, candidate_b_id: String },
    /// Source pairwise flags contradict each other.
    InconsistentPairClassification { candidate_a_id: String, candidate_b_id: String },
    /// An action's protocol revision differs from its source report.
    ProtocolMismatch { action_id: String },
    /// Action references an unknown or duplicate candidate ID.
    InvalidActionCandidateCoverage { action_id: String, candidate_id: String },
    /// Action contains no pair to compare.
    InsufficientActionCandidates { action_id: String },
    /// Action lacks target metrics or repeats a metric target.
    InvalidActionMetricTargets { action_id: String },
    /// Cost interval is not finite, non-negative or ordered.
    InvalidCostInterval { action_id: String },
}

/// Evaluate declared measurement actions against dominance-sensitive cells in one source report.
///
/// The only eligible comparison cells are pairs classified as
/// `additional_evidence_may_change_dominance` and dimensions not separated throughout their
/// declared intervals. Definite trade-offs are not sent to "measure more and a winner will
/// emerge": they require an explicit user/task preference or a hard constraint.
///
/// Eligible actions form a Pareto set under two transparent criteria:
/// (1) coverage must be a superset of the dominated action's decision-sensitive cells, and
/// (2) the covering action's conservative upper cost bound must be no greater. A larger but
/// differently targeted action does not erase a smaller action that covers a different cell.
///
/// Caller-declared gate statuses are not authenticated. This function creates proposals only;
/// it never runs an instrument, authorizes a human study, or emits actuator commands.
pub fn triage_measurement_actions(
    source: &ArchitectureRobustnessReportV1,
    actions: &[MeasurementActionV1],
    schema_version: u16,
) -> Result<MeasurementTriageReportV1, Vec<MeasurementTriageFindingV1>> {
    let mut findings = Vec::new();
    if schema_version != MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1 {
        findings.push(MeasurementTriageFindingV1::UnsupportedSchemaVersion {
            received: schema_version,
        });
    }

    validate_source_report(source, &mut findings);

    let candidate_ids: HashSet<&str> = source
        .candidates
        .iter()
        .map(|candidate| candidate.candidate_id.as_str())
        .collect();
    let mut action_ids = HashSet::with_capacity(actions.len());

    for action in actions {
        if action.action_id.trim().is_empty() || action.action_id.trim() != action.action_id {
            findings.push(MeasurementTriageFindingV1::InvalidOrDuplicateIdentity {
                identity: action.action_id.clone(),
            });
        }
        if !action_ids.insert(action.action_id.as_str()) {
            findings.push(MeasurementTriageFindingV1::InvalidOrDuplicateIdentity {
                identity: action.action_id.clone(),
            });
        }
        if action.task_protocol_version != source.task_protocol_version {
            findings.push(MeasurementTriageFindingV1::ProtocolMismatch {
                action_id: action.action_id.clone(),
            });
        }
        if action.candidate_ids.len() < 2 {
            findings.push(MeasurementTriageFindingV1::InsufficientActionCandidates {
                action_id: action.action_id.clone(),
            });
        }
        let mut seen_candidates = HashSet::with_capacity(action.candidate_ids.len());
        for candidate_id in &action.candidate_ids {
            if !candidate_ids.contains(candidate_id.as_str())
                || !seen_candidates.insert(candidate_id.as_str())
            {
                findings.push(MeasurementTriageFindingV1::InvalidActionCandidateCoverage {
                    action_id: action.action_id.clone(),
                    candidate_id: candidate_id.clone(),
                });
            }
        }
        let mut seen_metrics = HashSet::with_capacity(action.metric_targets.len());
        if action.metric_targets.is_empty()
            || action.metric_targets.iter().any(|metric| !seen_metrics.insert(*metric))
        {
            findings.push(MeasurementTriageFindingV1::InvalidActionMetricTargets {
                action_id: action.action_id.clone(),
            });
        }
        let cost = action.cost_minutes;
        if !cost.lower_minutes.is_finite()
            || !cost.upper_minutes.is_finite()
            || cost.lower_minutes < 0.0
            || cost.upper_minutes < cost.lower_minutes
        {
            findings.push(MeasurementTriageFindingV1::InvalidCostInterval {
                action_id: action.action_id.clone(),
            });
        }
    }

    if !findings.is_empty() {
        return Err(findings);
    }

    let measurement_sensitive_pairs: Vec<_> = source
        .pairwise_assessments
        .iter()
        .filter(|pair| pair.additional_evidence_may_change_dominance)
        .collect();

    let decision_sensitive_cell_count: usize = measurement_sensitive_pairs
        .iter()
        .map(|pair| pair.not_separated_dimensions.len())
        .sum();

    let all_cells: Vec<DecisionSensitiveCellV1> = measurement_sensitive_pairs
        .iter()
        .flat_map(|pair| {
            pair.not_separated_dimensions.iter().map(|metric| DecisionSensitiveCellV1 {
                candidate_a_id: pair.candidate_a_id.clone(),
                candidate_b_id: pair.candidate_b_id.clone(),
                metric: *metric,
            })
        })
        .collect();

    let mut assessments = Vec::with_capacity(actions.len());
    for action in actions {
        let mut blockers = evaluate_action_gates(action);
        let covered_cells: Vec<DecisionSensitiveCellV1> = all_cells
            .iter()
            .filter(|cell| {
                action.metric_targets.contains(&cell.metric)
                    && action.candidate_ids.contains(&cell.candidate_a_id)
                    && action.candidate_ids.contains(&cell.candidate_b_id)
            })
            .cloned()
            .collect();
        let covered_cell_count = covered_cells.len();

        if covered_cell_count == 0 {
            blockers.push(MeasurementActionBlockerV1 {
                gate: "decision_relevance".to_owned(),
                status: MeasurementGateStatusV1::NotApplicable,
                reason: "Action covers no metric-pair cell whose dominance may change within the source intervals; it cannot enter this decision-sensitive frontier.".to_owned(),
            });
        }

        assessments.push(MeasurementActionAssessmentV1 {
            action_id: action.action_id.clone(),
            kind: action.kind,
            eligible_for_triage: blockers.is_empty(),
            blockers,
            covered_cells,
            covered_cell_count,
            on_proposal_frontier: false,
            dominated_by: Vec::new(),
            cost_minutes: action.cost_minutes,
        });
    }

    for index in 0..assessments.len() {
        if !assessments[index].eligible_for_triage {
            continue;
        }
        let mut dominated_by = Vec::new();
        for other_index in 0..assessments.len() {
            if index == other_index || !assessments[other_index].eligible_for_triage {
                continue;
            }
            if action_dominates(&assessments[other_index], &assessments[index]) {
                dominated_by.push(assessments[other_index].action_id.clone());
            }
        }
        assessments[index].dominated_by = dominated_by;
        assessments[index].on_proposal_frontier = assessments[index].dominated_by.is_empty();
    }

    Ok(MeasurementTriageReportV1 {
        schema_version: MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
        comparison_id: source.comparison_id.clone(),
        task_protocol_version: source.task_protocol_version.clone(),
        evidence_level: source.evidence_level,
        measurement_sensitive_pair_count: measurement_sensitive_pairs.len(),
        decision_sensitive_cell_count,
        actions: assessments,
        assumptions: vec![
            "Action coverage is caller-declared and is not independently verified.".to_owned(),
            "Covered metric-pair cells are deterministic diagnostics, not expected information gain.".to_owned(),
            "A measurement can narrow bounds without resolving dominance; no resolution is guaranteed.".to_owned(),
            "Definite trade-offs are not sent to a dominance-resolution queue.".to_owned(),
            "Gate statuses are caller-declared and are not authenticated.".to_owned(),
            "No physical or human-participant action is executed or authorized by this report.".to_owned(),
        ],
        claim_ceiling: "Triage of caller-declared measurement proposals against supplied interval comparison evidence only; no experiment outcome, clinical benefit, safety or execution authority.".to_owned(),
    })
}

fn validate_source_report(
    source: &ArchitectureRobustnessReportV1,
    findings: &mut Vec<MeasurementTriageFindingV1>,
) {
    if source.schema_version != ARCHITECTURE_ROBUSTNESS_SCHEMA_VERSION_V1 {
        findings.push(
            MeasurementTriageFindingV1::UnsupportedSourceSchemaVersion {
                received: source.schema_version,
            },
        );
    }
    if source.candidates.len() < 2
        || source.comparison_id.trim().is_empty()
        || source.task_protocol_version.trim().is_empty()
    {
        findings.push(MeasurementTriageFindingV1::IncompleteSourceComparison);
    }

    let mut candidate_ids = HashSet::with_capacity(source.candidates.len());
    for candidate in &source.candidates {
        if candidate.candidate_id.trim().is_empty()
            || candidate.candidate_id.trim() != candidate.candidate_id
            || !candidate_ids.insert(candidate.candidate_id.as_str())
        {
            findings.push(MeasurementTriageFindingV1::InvalidOrDuplicateIdentity {
                identity: candidate.candidate_id.clone(),
            });
        }
    }

    let expected_pairs = source.candidates.len() * source.candidates.len().saturating_sub(1) / 2;
    if source.pairwise_assessments.len() != expected_pairs {
        findings.push(MeasurementTriageFindingV1::IncompleteSourceComparison);
    }

    let mut seen_pairs = HashSet::with_capacity(source.pairwise_assessments.len());
    for pair in &source.pairwise_assessments {
        if pair.candidate_a_id == pair.candidate_b_id
            || !candidate_ids.contains(pair.candidate_a_id.as_str())
            || !candidate_ids.contains(pair.candidate_b_id.as_str())
        {
            findings.push(MeasurementTriageFindingV1::InvalidOrDuplicatePair {
                candidate_a_id: pair.candidate_a_id.clone(),
                candidate_b_id: pair.candidate_b_id.clone(),
            });
            continue;
        }
        let key = if pair.candidate_a_id < pair.candidate_b_id {
            (pair.candidate_a_id.as_str(), pair.candidate_b_id.as_str())
        } else {
            (pair.candidate_b_id.as_str(), pair.candidate_a_id.as_str())
        };
        if !seen_pairs.insert(key) {
            findings.push(MeasurementTriageFindingV1::InvalidOrDuplicatePair {
                candidate_a_id: pair.candidate_a_id.clone(),
                candidate_b_id: pair.candidate_b_id.clone(),
            });
        }

        let expected_unresolved =
            !pair.a_robustly_dominates_b && !pair.b_robustly_dominates_a;
        let expected_tradeoff = !pair.a_definitely_better_dimensions.is_empty()
            && !pair.b_definitely_better_dimensions.is_empty();
        if (pair.a_robustly_dominates_b && pair.b_robustly_dominates_a)
            || pair.unresolved != expected_unresolved
            || pair.definite_tradeoff != expected_tradeoff
            || ((pair.a_robustly_dominates_b || pair.b_robustly_dominates_a)
                && (pair.definite_tradeoff || pair.additional_evidence_may_change_dominance))
            || (pair.definite_tradeoff
                && (!pair.unresolved
                    || pair.a_definitely_better_dimensions.is_empty()
                    || pair.b_definitely_better_dimensions.is_empty()))
            || (pair.additional_evidence_may_change_dominance
                && (!pair.unresolved
                    || pair.definite_tradeoff
                    || !(pair.a_may_dominate_b_within_bounds
                        || pair.b_may_dominate_a_within_bounds)))
        {
            findings.push(
                MeasurementTriageFindingV1::InconsistentPairClassification {
                    candidate_a_id: pair.candidate_a_id.clone(),
                    candidate_b_id: pair.candidate_b_id.clone(),
                },
            );
        }
    }
}

fn evaluate_action_gates(action: &MeasurementActionV1) -> Vec<MeasurementActionBlockerV1> {
    let mut blockers = Vec::new();
    for (name, status, is_required) in [
        ("protocol_frozen", action.gates.protocol_frozen, true),
        ("candidate_coverage", action.gates.candidate_coverage, true),
        ("feasibility_review", action.gates.feasibility_review, true),
        (
            "safety_review",
            action.gates.safety_review,
            matches!(
                action.kind,
                MeasurementActionKindV1::BenchMeasurement | MeasurementActionKindV1::HumanStudy
            ),
        ),
        (
            "consent_review",
            action.gates.consent_review,
            action.kind == MeasurementActionKindV1::HumanStudy,
        ),
        (
            "calibration_status",
            action.gates.calibration_status,
            matches!(
                action.kind,
                MeasurementActionKindV1::BenchMeasurement | MeasurementActionKindV1::HumanStudy
            ),
        ),
        ("evidence_provenance", action.gates.evidence_provenance, true),
    ] {
        let okay = if is_required {
            status == MeasurementGateStatusV1::Satisfied
        } else {
            matches!(
                status,
                MeasurementGateStatusV1::NotApplicable | MeasurementGateStatusV1::Satisfied
            )
        };
        if !okay {
            blockers.push(MeasurementActionBlockerV1 {
                gate: name.to_owned(),
                status,
                reason: if status == MeasurementGateStatusV1::Unknown {
                    "Unknown prerequisite status fails closed.".to_owned()
                } else if is_required {
                    "Required prerequisite must be explicitly satisfied for this action.".to_owned()
                } else {
                    "A non-applicable gate may be omitted only when explicitly marked not_applicable.".to_owned()
                },
            });
        }
    }
    blockers
}

fn action_dominates(a: &MeasurementActionAssessmentV1, b: &MeasurementActionAssessmentV1) -> bool {
    let cost_no_worse =
        a.cost_minutes.upper_minutes <= b.cost_minutes.upper_minutes + MEASUREMENT_TRIAGE_EPSILON_V1;
    if !cost_no_worse {
        return false;
    }
    let a_cells: HashSet<&DecisionSensitiveCellV1> = a.covered_cells.iter().collect();
    let b_cells: HashSet<&DecisionSensitiveCellV1> = b.covered_cells.iter().collect();
    let coverage_superset = b_cells.is_subset(&a_cells);
    let strict_cost =
        a.cost_minutes.upper_minutes < b.cost_minutes.upper_minutes - MEASUREMENT_TRIAGE_EPSILON_V1;
    coverage_superset && (strict_cost || a_cells.len() > b_cells.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_robustness::{
        ArchitectureRobustnessReportV1, MetricIntervalEntryV1, RobustCandidateAssessmentV1,
        RobustPairwiseAssessmentV1,
    };
    use crate::architecture_pareto::{
        ArchitectureFamilyV1, ArchitectureOperatingConditionsV1,
    };

    fn gates(kind: MeasurementActionKindV1) -> MeasurementActionGatesV1 {
        MeasurementActionGatesV1 {
            protocol_frozen: MeasurementGateStatusV1::Satisfied,
            candidate_coverage: MeasurementGateStatusV1::Satisfied,
            feasibility_review: MeasurementGateStatusV1::Satisfied,
            safety_review: if matches!(
                kind,
                MeasurementActionKindV1::BenchMeasurement | MeasurementActionKindV1::HumanStudy
            ) {
                MeasurementGateStatusV1::Satisfied
            } else {
                MeasurementGateStatusV1::NotApplicable
            },
            consent_review: if kind == MeasurementActionKindV1::HumanStudy {
                MeasurementGateStatusV1::Satisfied
            } else {
                MeasurementGateStatusV1::NotApplicable
            },
            calibration_status: if matches!(
                kind,
                MeasurementActionKindV1::BenchMeasurement | MeasurementActionKindV1::HumanStudy
            ) {
                MeasurementGateStatusV1::Satisfied
            } else {
                MeasurementGateStatusV1::NotApplicable
            },
            evidence_provenance: MeasurementGateStatusV1::Satisfied,
        }
    }

    fn candidate(id: &str) -> RobustCandidateAssessmentV1 {
        RobustCandidateAssessmentV1 {
            candidate_id: id.to_owned(),
            family: ArchitectureFamilyV1::HybridPoweredPassive,
            tethered: false,
            on_robust_frontier: true,
            robustly_dominated_by: Vec::new(),
            unresolved_against: Vec::new(),
            metrics: Vec::<MetricIntervalEntryV1>::new(),
        }
    }

    fn source_report() -> ArchitectureRobustnessReportV1 {
        let ambiguous = RobustPairwiseAssessmentV1 {
            candidate_a_id: "candidate-a".to_owned(),
            candidate_b_id: "candidate-b".to_owned(),
            a_robustly_dominates_b: false,
            b_robustly_dominates_a: false,
            a_may_dominate_b_within_bounds: true,
            b_may_dominate_a_within_bounds: true,
            not_separated_dimensions: vec![
                ArchitectureMetricV1::WearerBorneMassKg,
                ArchitectureMetricV1::EnergyWhPerTask,
            ],
            a_definitely_better_dimensions: Vec::new(),
            b_definitely_better_dimensions: Vec::new(),
            definite_tradeoff: false,
            additional_evidence_may_change_dominance: true,
            unresolved: true,
        };
        let tradeoff = RobustPairwiseAssessmentV1 {
            candidate_a_id: "candidate-a".to_owned(),
            candidate_b_id: "candidate-c".to_owned(),
            a_robustly_dominates_b: false,
            b_robustly_dominates_a: false,
            a_may_dominate_b_within_bounds: false,
            b_may_dominate_a_within_bounds: false,
            not_separated_dimensions: vec![ArchitectureMetricV1::ComfortScore],
            a_definitely_better_dimensions: vec![ArchitectureMetricV1::TaskSuccessRate],
            b_definitely_better_dimensions: vec![ArchitectureMetricV1::WearerBorneMassKg],
            definite_tradeoff: true,
            additional_evidence_may_change_dominance: false,
            unresolved: true,
        };
        let third_pair = RobustPairwiseAssessmentV1 {
            candidate_a_id: "candidate-b".to_owned(),
            candidate_b_id: "candidate-c".to_owned(),
            a_robustly_dominates_b: false,
            b_robustly_dominates_a: true,
            a_may_dominate_b_within_bounds: false,
            b_may_dominate_a_within_bounds: true,
            not_separated_dimensions: Vec::new(),
            a_definitely_better_dimensions: Vec::new(),
            b_definitely_better_dimensions: vec![ArchitectureMetricV1::TaskSuccessRate],
            definite_tradeoff: false,
            additional_evidence_may_change_dominance: false,
            unresolved: false,
        };
        ArchitectureRobustnessReportV1 {
            schema_version: 1,
            comparison_id: "triage-fixture-001".to_owned(),
            benchmark_case_id: "walking-case".to_owned(),
            task_id: "walk".to_owned(),
            wearer_profile_id: "profile-a".to_owned(),
            environment_profile_id: "flat-lab".to_owned(),
            task_protocol_version: "walk-v1".to_owned(),
            operating_conditions: ArchitectureOperatingConditionsV1 {
                speed_mps: 0.8,
                payload_kg: 0.0,
                grade_percent: 0.0,
                ambient_temperature_c: 22.0,
            },
            interval_semantics: crate::architecture_robustness::MetricIntervalSemanticsV1::DeclaredConservativeEnvelope,
            evidence_level: ArchitectureEvidenceLevelV1::SyntheticReplay,
            candidates: vec![candidate("candidate-a"), candidate("candidate-b"), candidate("candidate-c")],
            pairwise_assessments: vec![ambiguous, tradeoff, third_pair],
            assumptions: Vec::new(),
            claim_ceiling: "Synthetic triage fixture only".to_owned(),
        }
    }

    fn action(
        id: &str,
        candidates: &[&str],
        metrics: &[ArchitectureMetricV1],
        upper_minutes: f64,
        kind: MeasurementActionKindV1,
    ) -> MeasurementActionV1 {
        MeasurementActionV1 {
            action_id: id.to_owned(),
            kind,
            task_protocol_version: "walk-v1".to_owned(),
            candidate_ids: candidates.iter().map(|id| (*id).to_owned()).collect(),
            metric_targets: metrics.to_vec(),
            cost_minutes: MeasurementCostIntervalV1 {
                lower_minutes: upper_minutes * 0.5,
                upper_minutes,
            },
            gates: gates(kind),
        }
    }

    #[test]
    fn action_frontier_preserves_coverage_cost_tradeoffs() {
        let actions = vec![
            action(
                "measure-mass",
                &["candidate-a", "candidate-b"],
                &[ArchitectureMetricV1::WearerBorneMassKg],
                20.0,
                MeasurementActionKindV1::BenchMeasurement,
            ),
            action(
                "measure-energy",
                &["candidate-a", "candidate-b"],
                &[ArchitectureMetricV1::EnergyWhPerTask],
                18.0,
                MeasurementActionKindV1::BenchMeasurement,
            ),
            action(
                "measure-both",
                &["candidate-a", "candidate-b"],
                &[
                    ArchitectureMetricV1::WearerBorneMassKg,
                    ArchitectureMetricV1::EnergyWhPerTask,
                ],
                30.0,
                MeasurementActionKindV1::BenchMeasurement,
            ),
            action(
                "measure-mass-cheap",
                &["candidate-a", "candidate-b"],
                &[ArchitectureMetricV1::WearerBorneMassKg],
                10.0,
                MeasurementActionKindV1::BenchMeasurement,
            ),
        ];
        let report = triage_measurement_actions(
            &source_report(),
            &actions,
            MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
        )
        .expect("valid synthetic report/action catalog");
        let frontier: Vec<&str> = report
            .actions
            .iter()
            .filter(|assessment| assessment.on_proposal_frontier)
            .map(|assessment| assessment.action_id.as_str())
            .collect();
        assert_eq!(frontier, vec!["measure-energy", "measure-both", "measure-mass-cheap"]);
        assert_eq!(report.measurement_sensitive_pair_count, 1);
        assert_eq!(report.decision_sensitive_cell_count, 2);
        assert_eq!(report.actions[0].covered_cell_count, 1);
        assert_eq!(report.actions[2].covered_cell_count, 2);
    }

    #[test]
    fn definite_tradeoffs_are_not_misreported_as_measurement_sensitive() {
        let report = triage_measurement_actions(
            &source_report(),
            &[action(
                "measure-comfort",
                &["candidate-a", "candidate-c"],
                &[ArchitectureMetricV1::ComfortScore],
                5.0,
                MeasurementActionKindV1::BenchMeasurement,
            )],
            MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
        )
        .expect("valid fixture");
        assert_eq!(report.measurement_sensitive_pair_count, 1);
        assert_eq!(report.actions[0].covered_cell_count, 0);
        assert!(!report.actions[0].eligible_for_triage);
        assert!(!report.actions[0].on_proposal_frontier);
        assert!(report.actions[0].blockers.iter().any(|blocker| {
            blocker.gate == "decision_relevance"
        }));
    }

    #[test]
    fn unknown_safety_consent_or_calibration_fails_closed() {
        for (kind, gate_name) in [
            (MeasurementActionKindV1::BenchMeasurement, "calibration_status"),
            (MeasurementActionKindV1::BenchMeasurement, "safety_review"),
            (MeasurementActionKindV1::HumanStudy, "consent_review"),
        ] {
            let mut candidate_action = action(
                "candidate-action",
                &["candidate-a", "candidate-b"],
                &[ArchitectureMetricV1::WearerBorneMassKg],
                10.0,
                kind,
            );
            match gate_name {
                "calibration_status" => {
                    candidate_action.gates.calibration_status = MeasurementGateStatusV1::Unknown
                }
                "safety_review" => {
                    candidate_action.gates.safety_review = MeasurementGateStatusV1::Unknown
                }
                "consent_review" => {
                    candidate_action.gates.consent_review = MeasurementGateStatusV1::Unknown
                }
                _ => unreachable!(),
            }
            let report = triage_measurement_actions(
                &source_report(),
                &[candidate_action],
                MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
            )
            .expect("unknown gate is a blocked proposal, not malformed input");
            assert!(!report.actions[0].eligible_for_triage);
            assert!(!report.actions[0].on_proposal_frontier);
            assert!(report.actions[0]
                .blockers
                .iter()
                .any(|blocker| blocker.gate == gate_name));
        }
    }

    #[test]
    fn stale_protocol_revision_is_rejected() {
        let mut candidate_action = action(
            "stale",
            &["candidate-a", "candidate-b"],
            &[ArchitectureMetricV1::WearerBorneMassKg],
            10.0,
            MeasurementActionKindV1::BenchMeasurement,
        );
        candidate_action.task_protocol_version = "walk-v0".to_owned();
        assert!(
            triage_measurement_actions(
                &source_report(),
                &[candidate_action],
                MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
            )
            .as_ref()
            .err()
            .unwrap()
            .iter()
            .any(|finding| matches!(
                finding,
                MeasurementTriageFindingV1::ProtocolMismatch { .. }
            ))
        );
    }

    #[test]
    fn duplicate_metric_candidate_or_action_ids_fail_closed() {
        let mut duplicate_metric = action(
            "duplicate",
            &["candidate-a", "candidate-b"],
            &[ArchitectureMetricV1::WearerBorneMassKg],
            10.0,
            MeasurementActionKindV1::BenchMeasurement,
        );
        duplicate_metric.metric_targets.push(ArchitectureMetricV1::WearerBorneMassKg);
        assert!(triage_measurement_actions(
            &source_report(),
            &[duplicate_metric],
            MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
        )
        .is_err());

        let repeated = action(
            "same-id",
            &["candidate-a", "candidate-b"],
            &[ArchitectureMetricV1::WearerBorneMassKg],
            10.0,
            MeasurementActionKindV1::BenchMeasurement,
        );
        let repeated2 = action(
            "same-id",
            &["candidate-a", "candidate-b"],
            &[ArchitectureMetricV1::EnergyWhPerTask],
            11.0,
            MeasurementActionKindV1::BenchMeasurement,
        );
        assert!(triage_measurement_actions(
            &source_report(),
            &[repeated, repeated2],
            MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
        )
        .is_err());
    }

    #[test]
    fn same_catalog_produces_byte_stable_report() {
        let source = source_report();
        let actions = vec![
            action(
                "measure-mass",
                &["candidate-a", "candidate-b"],
                &[ArchitectureMetricV1::WearerBorneMassKg],
                20.0,
                MeasurementActionKindV1::BenchMeasurement,
            ),
            action(
                "measure-energy",
                &["candidate-a", "candidate-b"],
                &[ArchitectureMetricV1::EnergyWhPerTask],
                18.0,
                MeasurementActionKindV1::BenchMeasurement,
            ),
        ];
        let expected = triage_measurement_actions(
            &source,
            &actions,
            MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
        )
        .expect("valid fixture");
        let bytes = serde_json::to_vec(&expected).expect("report serializes");
        for _ in 0..100 {
            let actual = triage_measurement_actions(
                &source,
                &actions,
                MEASUREMENT_TRIAGE_SCHEMA_VERSION_V1,
            )
            .expect("valid fixture");
            assert_eq!(actual, expected);
            assert_eq!(serde_json::to_vec(&actual).expect("report serializes"), bytes);
        }
    }
}
