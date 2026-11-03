// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-qualified traffic-rule compliance evaluation.
//!
//! This module evaluates supplied rule evidence against a supplied trajectory.
//! It does not perform perception, map matching, sign recognition, traffic-law
//! interpretation or vehicle simulation. Missing/contradictory rule evidence
//! yields Indeterminate, never an assumed-compliant result.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

pub const TRAFFIC_RULE_COMPLIANCE_SCHEMA: &str =
    "symthaea.vehicle.traffic-rule-compliance.v0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrafficRuleEvidenceStatus {
    /// The supplied evidence source asserts this rule without a recorded conflict.
    /// This does not authenticate the source or prove perception correctness.
    Uncontradicted,
    /// Available sources disagree or the rule interpretation is unresolved.
    Conflicted,
    /// No usable rule evidence was provided.
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrafficRuleEvidence {
    pub evidence_id: String,
    pub source_id: String,
    pub source_artifact_digest: String,
    pub status: TrafficRuleEvidenceStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrafficRuleKind {
    SpeedLimit {
        maximum_speed_mps: f64,
        tolerance_mps: f64,
        maximum_sample_gap_s: f64,
    },
    MandatoryDirection {
        required_heading_rad: f64,
        maximum_heading_error_rad: f64,
        maximum_sample_gap_s: f64,
    },
    StopBeforeLine {
        stop_line_progress_m: f64,
        stop_speed_threshold_mps: f64,
        required_stationary_duration_s: f64,
        maximum_sample_gap_s: f64,
        progress_tolerance_m: f64,
    },
    /// A declared family for which this verifier has no executable semantics.
    Unsupported { rule_family: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrafficRuleRequirement {
    pub rule_id: String,
    pub active_start_progress_m: f64,
    pub active_end_progress_m: f64,
    pub kind: TrafficRuleKind,
    pub evidence: TrafficRuleEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RuleTrajectorySample {
    pub timestamp_s: f64,
    /// Monotonic route progress in meters; loop-wrap must be unwrapped by caller.
    pub route_progress_m: f64,
    pub speed_mps: f64,
    pub heading_rad: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleComplianceOutcome {
    Compliant,
    Violation,
    Indeterminate,
    NotEncountered,
    Pending,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleComplianceReason {
    EvidenceUnavailable,
    EvidenceConflicted,
    EvidenceMalformed,
    RuleMalformed,
    TrajectoryMalformed,
    TrajectorySamplingGap,
    TrajectoryCoverageInsufficient,
    TrajectoryKinematicInconsistency,
    SpeedLimitExceeded,
    MandatoryDirectionNotFollowed,
    StopLinePassedWithoutRequiredStop,
    RequirementSatisfied,
    RuleKindUnsupported,
    RequiredStopObserved,
    RequirementNotEncountered,
    RequirementPending,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleComplianceFinding {
    pub rule_id: String,
    pub evidence_id: String,
    pub outcome: RuleComplianceOutcome,
    pub reason: RuleComplianceReason,
    pub applicable_sample_count: usize,
    pub first_violating_sample_index: Option<usize>,
    pub observed_minimum_speed_mps: Option<f64>,
    pub observed_maximum_speed_mps: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleEvaluationError {
    EmptyTrajectory,
    DuplicateRuleId,
    EmptyRuleId,
    NonFiniteInput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrafficRuleComplianceReport {
    pub schema_version: String,
    pub requirements: Vec<TrafficRuleRequirement>,
    pub trajectory: Vec<RuleTrajectorySample>,
    pub findings: Vec<RuleComplianceFinding>,
    pub compliant_count: usize,
    pub violation_count: usize,
    pub indeterminate_count: usize,
    pub not_encountered_count: usize,
    pub pending_count: usize,
    /// BLAKE3 over exact inputs, findings and aggregates with this field empty.
    pub report_digest: String,
}

impl TrafficRuleComplianceReport {
    /// Re-evaluate retained rule/trajectory inputs and reconstruct all aggregates.
    pub fn identity_is_valid(&self) -> bool {
        if self.schema_version != TRAFFIC_RULE_COMPLIANCE_SCHEMA
            || self.trajectory.is_empty()
            || self.requirements.len() != self.findings.len()
            || self.report_digest.len() != 64
            || !self.report_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return false;
        }

        let mut rule_ids = BTreeSet::new();
        if self.requirements.iter().any(|requirement| {
            requirement.rule_id.trim().is_empty()
                || !rule_ids.insert(requirement.rule_id.as_str())
        }) {
            return false;
        }

        let expected_findings = self
            .requirements
            .iter()
            .map(|requirement| evaluate_requirement(requirement, &self.trajectory))
            .collect::<Vec<_>>();
        if expected_findings != self.findings {
            return false;
        }

        let counts = outcome_counts(&self.findings);
        if self.compliant_count != counts.0
            || self.violation_count != counts.1
            || self.indeterminate_count != counts.2
            || self.not_encountered_count != counts.3
            || self.pending_count != counts.4
        {
            return false;
        }

        let mut report = self.clone();
        let expected_digest = report.report_digest.clone();
        report.report_digest.clear();
        serde_json::to_vec(&report)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected_digest)
            .unwrap_or(false)
    }
}

/// Evaluate rule records against an exact route-progress trajectory.
///
/// This is an offline compliance checker. The caller is responsible for
/// supplying correctly localized rule positions and trajectory measurements.
/// The report retains these raw inputs so an independent checker can replay
/// the rule evaluation.
pub fn evaluate_traffic_rules(
    requirements: &[TrafficRuleRequirement],
    trajectory: &[RuleTrajectorySample],
) -> Result<TrafficRuleComplianceReport, RuleEvaluationError> {
    if trajectory.is_empty() {
        return Err(RuleEvaluationError::EmptyTrajectory);
    }
    if trajectory.iter().any(|sample| {
        !sample.timestamp_s.is_finite()
            || !sample.route_progress_m.is_finite()
            || !sample.speed_mps.is_finite()
            || !sample.heading_rad.is_finite()
    }) || requirements.iter().any(|requirement| !requirement_numeric_fields_are_finite(requirement)) {
        return Err(RuleEvaluationError::NonFiniteInput);
    }

    let mut rule_ids = BTreeSet::new();
    for requirement in requirements {
        if requirement.rule_id.trim().is_empty() {
            return Err(RuleEvaluationError::EmptyRuleId);
        }
        if !rule_ids.insert(requirement.rule_id.as_str()) {
            return Err(RuleEvaluationError::DuplicateRuleId);
        }
    }

    let findings = requirements
        .iter()
        .map(|requirement| evaluate_requirement(requirement, trajectory))
        .collect::<Vec<_>>();
    let counts = outcome_counts(&findings);

    let mut report = TrafficRuleComplianceReport {
        schema_version: TRAFFIC_RULE_COMPLIANCE_SCHEMA.into(),
        requirements: requirements.to_vec(),
        trajectory: trajectory.to_vec(),
        findings,
        compliant_count: counts.0,
        violation_count: counts.1,
        indeterminate_count: counts.2,
        not_encountered_count: counts.3,
        pending_count: counts.4,
        report_digest: String::new(),
    };
    let bytes = serde_json::to_vec(&report).map_err(|_| RuleEvaluationError::EmptyTrajectory)?;
    report.report_digest = blake3::hash(&bytes).to_hex().to_string();
    Ok(report)
}

fn evaluate_requirement(
    requirement: &TrafficRuleRequirement,
    trajectory: &[RuleTrajectorySample],
) -> RuleComplianceFinding {
    let indeterminate = |reason| RuleComplianceFinding {
        rule_id: requirement.rule_id.clone(),
        evidence_id: requirement.evidence.evidence_id.clone(),
        outcome: RuleComplianceOutcome::Indeterminate,
        reason,
        applicable_sample_count: 0,
        first_violating_sample_index: None,
        observed_minimum_speed_mps: None,
        observed_maximum_speed_mps: None,
    };

    if !trajectory_is_valid(trajectory) {
        return indeterminate(RuleComplianceReason::TrajectoryMalformed);
    }
    if !evidence_is_well_formed(&requirement.evidence) {
        return indeterminate(RuleComplianceReason::EvidenceMalformed);
    }
    match requirement.evidence.status {
        TrafficRuleEvidenceStatus::Unavailable => {
            return indeterminate(RuleComplianceReason::EvidenceUnavailable);
        }
        TrafficRuleEvidenceStatus::Conflicted => {
            return indeterminate(RuleComplianceReason::EvidenceConflicted);
        }
        TrafficRuleEvidenceStatus::Uncontradicted => {}
    }

    if requirement.rule_id.trim().is_empty()
        || !requirement.active_start_progress_m.is_finite()
        || !requirement.active_end_progress_m.is_finite()
        || requirement.active_start_progress_m > requirement.active_end_progress_m
    {
        return indeterminate(RuleComplianceReason::RuleMalformed);
    }

    match &requirement.kind {
        TrafficRuleKind::SpeedLimit {
            maximum_speed_mps,
            tolerance_mps,
            maximum_sample_gap_s,
        } => {
            if !maximum_speed_mps.is_finite()
                || maximum_speed_mps <= 0.0
                || !tolerance_mps.is_finite()
                || tolerance_mps < 0.0
                || !maximum_sample_gap_s.is_finite()
                || *maximum_sample_gap_s <= 0.0
            {
                return indeterminate(RuleComplianceReason::RuleMalformed);
            }
            let applicable = trajectory
                .iter()
                .enumerate()
                .filter(|(_, sample)| {
                    sample.route_progress_m >= requirement.active_start_progress_m
                        && sample.route_progress_m <= requirement.active_end_progress_m
                })
                .collect::<Vec<_>>();
            if !trajectory_range_overlaps_rule(trajectory, requirement) {
                return not_encountered(requirement);
            }
            if !trajectory_covers_rule(trajectory, requirement) {
                return indeterminate(RuleComplianceReason::TrajectoryCoverageInsufficient);
            }
            if applicable.len() < 2 {
                return indeterminate(RuleComplianceReason::TrajectoryCoverageInsufficient);
            }
            if rule_zone_has_sampling_gap(
                trajectory,
                requirement.active_start_progress_m,
                requirement.active_end_progress_m,
                *maximum_sample_gap_s,
            ) {
                return indeterminate(RuleComplianceReason::TrajectorySamplingGap);
            }
            let minimum = applicable
                .iter()
                .map(|(_, sample)| sample.speed_mps)
                .fold(f64::INFINITY, f64::min);
            let maximum = applicable
                .iter()
                .map(|(_, sample)| sample.speed_mps)
                .fold(f64::NEG_INFINITY, f64::max);
            let violating = applicable.iter().find(|(_, sample)| {
                sample.speed_mps > *maximum_speed_mps + *tolerance_mps
            });
            match violating {
                Some((index, _)) => finding(
                    requirement,
                    RuleComplianceOutcome::Violation,
                    RuleComplianceReason::SpeedLimitExceeded,
                    applicable.len(),
                    Some(*index),
                    Some(minimum),
                    Some(maximum),
                ),
                None => finding(
                    requirement,
                    RuleComplianceOutcome::Compliant,
                    RuleComplianceReason::RequirementSatisfied,
                    applicable.len(),
                    None,
                    Some(minimum),
                    Some(maximum),
                ),
            }
        }
        TrafficRuleKind::MandatoryDirection {
            required_heading_rad,
            maximum_heading_error_rad,
            maximum_sample_gap_s,
        } => {
            if !required_heading_rad.is_finite()
                || !maximum_heading_error_rad.is_finite()
                || !(0.0..=std::f64::consts::PI).contains(maximum_heading_error_rad)
                || !maximum_sample_gap_s.is_finite()
                || *maximum_sample_gap_s <= 0.0
            {
                return indeterminate(RuleComplianceReason::RuleMalformed);
            }
            let applicable = trajectory
                .iter()
                .enumerate()
                .filter(|(_, sample)| {
                    sample.route_progress_m >= requirement.active_start_progress_m
                        && sample.route_progress_m <= requirement.active_end_progress_m
                })
                .collect::<Vec<_>>();
            if !trajectory_range_overlaps_rule(trajectory, requirement) {
                return not_encountered(requirement);
            }
            if applicable.len() < 2 {
                return indeterminate(RuleComplianceReason::TrajectoryCoverageInsufficient);
            }
            if rule_zone_has_sampling_gap(
                trajectory,
                requirement.active_start_progress_m,
                requirement.active_end_progress_m,
                *maximum_sample_gap_s,
            ) {
                return indeterminate(RuleComplianceReason::TrajectorySamplingGap);
            }
            let violating = applicable.iter().find(|(_, sample)| {
                wrap_angle(sample.heading_rad - *required_heading_rad).abs()
                    > *maximum_heading_error_rad
            });
            match violating {
                Some((index, _)) => finding(
                    requirement,
                    RuleComplianceOutcome::Violation,
                    RuleComplianceReason::MandatoryDirectionNotFollowed,
                    applicable.len(),
                    Some(*index),
                    None,
                    None,
                ),
                None => finding(
                    requirement,
                    RuleComplianceOutcome::Compliant,
                    RuleComplianceReason::RequirementSatisfied,
                    applicable.len(),
                    None,
                    None,
                    None,
                ),
            }
        }
        TrafficRuleKind::Unsupported { rule_family } => {
            if rule_family.trim().is_empty() {
                indeterminate(RuleComplianceReason::RuleMalformed)
            } else {
                indeterminate(RuleComplianceReason::RuleKindUnsupported)
            }
        }
        TrafficRuleKind::StopBeforeLine {
            stop_line_progress_m,
            stop_speed_threshold_mps,
            required_stationary_duration_s,
            maximum_sample_gap_s,
            progress_tolerance_m,
        } => {
            if !stop_line_progress_m.is_finite()
                || *stop_line_progress_m < requirement.active_start_progress_m
                || *stop_line_progress_m > requirement.active_end_progress_m
                || !stop_speed_threshold_mps.is_finite()
                || *stop_speed_threshold_mps < 0.0
                || !required_stationary_duration_s.is_finite()
                || *required_stationary_duration_s < 0.0
                || !maximum_sample_gap_s.is_finite()
                || *maximum_sample_gap_s <= 0.0
                || !progress_tolerance_m.is_finite()
                || *progress_tolerance_m < 0.0
            {
                return indeterminate(RuleComplianceReason::RuleMalformed);
            }
            let crossing_index = trajectory.iter().position(|sample| {
                sample.route_progress_m > *stop_line_progress_m + *progress_tolerance_m
            });
            let Some(crossing_index) = crossing_index else {
                if trajectory.first().is_some_and(|first| {
                    first.route_progress_m > requirement.active_start_progress_m
                }) {
                    return indeterminate(RuleComplianceReason::TrajectoryCoverageInsufficient);
                }
                let active_zone_end = *stop_line_progress_m + *progress_tolerance_m;
                let overlaps_control_zone = trajectory.last().is_some_and(|last| {
                    last.route_progress_m >= requirement.active_start_progress_m
                }) && trajectory.first().is_some_and(|first| {
                    first.route_progress_m <= active_zone_end
                });
                if !overlaps_control_zone {
                    return not_encountered(requirement);
                }
                let active_sample_count = trajectory
                    .iter()
                    .filter(|sample| {
                        sample.route_progress_m >= requirement.active_start_progress_m
                            && sample.route_progress_m <= *stop_line_progress_m + *progress_tolerance_m
                    })
                    .count();
                if active_sample_count == 0 {
                    return indeterminate(RuleComplianceReason::TrajectoryCoverageInsufficient);
                }
                return finding(
                    requirement,
                    RuleComplianceOutcome::Pending,
                    RuleComplianceReason::RequirementPending,
                    active_sample_count,
                    None,
                    None,
                    None,
                );
            };
            if trajectory.first().is_some_and(|first| {
                first.route_progress_m > requirement.active_start_progress_m
            }) {
                return indeterminate(RuleComplianceReason::TrajectoryCoverageInsufficient);
            }
            let approach_zone_end = *stop_line_progress_m + *progress_tolerance_m;
            let approach_samples = trajectory[..=crossing_index]
                .iter()
                .filter(|sample| {
                    sample.route_progress_m >= requirement.active_start_progress_m
                        && sample.route_progress_m <= approach_zone_end
                })
                .count();
            if approach_samples < 2 {
                return indeterminate(RuleComplianceReason::TrajectoryCoverageInsufficient);
            }
            if rule_zone_has_sampling_gap(
                &trajectory[..=crossing_index],
                requirement.active_start_progress_m,
                approach_zone_end,
                *maximum_sample_gap_s,
            ) {
                return indeterminate(RuleComplianceReason::TrajectorySamplingGap);
            }
            let crossing_sample = &trajectory[crossing_index];
            let mut stop_start: Option<usize> = None;
            let mut index = crossing_index;
            let mut later_timestamp = crossing_sample.timestamp_s;
            let mut later_stopped_progress: Option<f64> = None;
            let mut latest_stopped_timestamp: Option<f64> = None;
            let mut insufficient_evidence_reason: Option<RuleComplianceReason> = None;
            while index > 0 {
                let previous = index - 1;
                let sample = &trajectory[previous];
                let sample_gap = later_timestamp - sample.timestamp_s;
                let progress_change_is_plausible = later_stopped_progress
                    .map(|later_progress| {
                        let observed_delta = later_progress - sample.route_progress_m;
                        observed_delta
                            <= *stop_speed_threshold_mps * sample_gap + *progress_tolerance_m
                    })
                    .unwrap_or(true);
                if sample_gap > *maximum_sample_gap_s {
                    insufficient_evidence_reason =
                        Some(RuleComplianceReason::TrajectorySamplingGap);
                    break;
                }
                if !progress_change_is_plausible {
                    insufficient_evidence_reason =
                        Some(RuleComplianceReason::TrajectoryKinematicInconsistency);
                    break;
                }
                if sample.route_progress_m > *stop_line_progress_m + *progress_tolerance_m
                    || sample.speed_mps > *stop_speed_threshold_mps
                {
                    break;
                }
                stop_start = Some(previous);
                if latest_stopped_timestamp.is_none() {
                    latest_stopped_timestamp = Some(sample.timestamp_s);
                }
                later_timestamp = sample.timestamp_s;
                later_stopped_progress = Some(sample.route_progress_m);
                index = previous;
            }
            let observed_dwell = stop_start
                .zip(latest_stopped_timestamp)
                .map(|(start, latest)| latest - trajectory[start].timestamp_s)
                .unwrap_or(0.0);
            if stop_start.is_some()
                && observed_dwell + 1e-9 >= *required_stationary_duration_s
            {
                finding(
                    requirement,
                    RuleComplianceOutcome::Compliant,
                    RuleComplianceReason::RequiredStopObserved,
                    crossing_index + 1,
                    None,
                    Some(
                        trajectory[stop_start.unwrap()..crossing_index]
                            .iter()
                            .map(|sample| sample.speed_mps)
                            .fold(f64::INFINITY, f64::min),
                    ),
                    Some(crossing_sample.speed_mps),
                )
            } else if let Some(reason) = insufficient_evidence_reason {
                indeterminate(reason)
            } else {
                finding(
                    requirement,
                    RuleComplianceOutcome::Violation,
                    RuleComplianceReason::StopLinePassedWithoutRequiredStop,
                    crossing_index + 1,
                    Some(crossing_index),
                    None,
                    Some(crossing_sample.speed_mps),
                )
            }
        }
    }
}

fn finding(
    requirement: &TrafficRuleRequirement,
    outcome: RuleComplianceOutcome,
    reason: RuleComplianceReason,
    applicable_sample_count: usize,
    first_violating_sample_index: Option<usize>,
    observed_minimum_speed_mps: Option<f64>,
    observed_maximum_speed_mps: Option<f64>,
) -> RuleComplianceFinding {
    RuleComplianceFinding {
        rule_id: requirement.rule_id.clone(),
        evidence_id: requirement.evidence.evidence_id.clone(),
        outcome,
        reason,
        applicable_sample_count,
        first_violating_sample_index,
        observed_minimum_speed_mps,
        observed_maximum_speed_mps,
    }
}

fn not_encountered(requirement: &TrafficRuleRequirement) -> RuleComplianceFinding {
    finding(
        requirement,
        RuleComplianceOutcome::NotEncountered,
        RuleComplianceReason::RequirementNotEncountered,
        0,
        None,
        None,
        None,
    )
}

fn requirement_numeric_fields_are_finite(requirement: &TrafficRuleRequirement) -> bool {
    if !requirement.active_start_progress_m.is_finite()
        || !requirement.active_end_progress_m.is_finite()
    {
        return false;
    }
    match &requirement.kind {
        TrafficRuleKind::SpeedLimit {
            maximum_speed_mps,
            tolerance_mps,
            maximum_sample_gap_s,
        } => maximum_speed_mps.is_finite()
            && tolerance_mps.is_finite()
            && maximum_sample_gap_s.is_finite(),
        TrafficRuleKind::MandatoryDirection {
            required_heading_rad,
            maximum_heading_error_rad,
            maximum_sample_gap_s,
        } => {
            required_heading_rad.is_finite()
                && maximum_heading_error_rad.is_finite()
                && maximum_sample_gap_s.is_finite()
        },
        TrafficRuleKind::Unsupported { .. } => true,
        TrafficRuleKind::StopBeforeLine {
            stop_line_progress_m,
            stop_speed_threshold_mps,
            required_stationary_duration_s,
            maximum_sample_gap_s,
            progress_tolerance_m,
        } => [
            stop_line_progress_m,
            stop_speed_threshold_mps,
            required_stationary_duration_s,
            maximum_sample_gap_s,
            progress_tolerance_m,
        ]
        .iter()
        .all(|value| value.is_finite()),
    }
}

fn trajectory_range_overlaps_rule(
    trajectory: &[RuleTrajectorySample],
    requirement: &TrafficRuleRequirement,
) -> bool {
    let Some(first) = trajectory.first() else {
        return false;
    };
    let Some(last) = trajectory.last() else {
        return false;
    };
    last.route_progress_m >= requirement.active_start_progress_m
        && first.route_progress_m <= requirement.active_end_progress_m
}

fn trajectory_covers_rule(
    trajectory: &[RuleTrajectorySample],
    requirement: &TrafficRuleRequirement,
) -> bool {
    let Some(first) = trajectory.first() else {
        return false;
    };
    let Some(last) = trajectory.last() else {
        return false;
    };
    first.route_progress_m <= requirement.active_start_progress_m
        && last.route_progress_m >= requirement.active_end_progress_m
}

fn rule_zone_has_sampling_gap(
    trajectory: &[RuleTrajectorySample],
    active_start_progress_m: f64,
    active_end_progress_m: f64,
    maximum_sample_gap_s: f64,
) -> bool {
    trajectory.windows(2).any(|pair| {
        let segment_overlaps_zone = pair[1].route_progress_m >= active_start_progress_m
            && pair[0].route_progress_m <= active_end_progress_m;
        segment_overlaps_zone
            && pair[1].timestamp_s - pair[0].timestamp_s > maximum_sample_gap_s
    })
}

fn evidence_is_well_formed(evidence: &TrafficRuleEvidence) -> bool {
    if evidence.evidence_id.trim().is_empty() {
        return false;
    }
    if evidence.status == TrafficRuleEvidenceStatus::Unavailable {
        // The absence of a usable source is itself the declared state.
        // No invented source ID or digest is required.
        return true;
    }
    !evidence.source_id.trim().is_empty()
        && evidence.source_artifact_digest.len() == 64
        && evidence
            .source_artifact_digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
}

fn trajectory_is_valid(trajectory: &[RuleTrajectorySample]) -> bool {
    if trajectory.is_empty() {
        return false;
    }
    trajectory.iter().all(|sample| {
        sample.timestamp_s.is_finite()
            && sample.route_progress_m.is_finite()
            && sample.speed_mps.is_finite()
            && sample.speed_mps >= 0.0
            && sample.heading_rad.is_finite()
    }) && trajectory.windows(2).all(|pair| {
        pair[1].timestamp_s > pair[0].timestamp_s
            && pair[1].route_progress_m >= pair[0].route_progress_m
    })
}

fn outcome_counts(findings: &[RuleComplianceFinding]) -> (usize, usize, usize, usize, usize) {
    let mut compliant = 0;
    let mut violation = 0;
    let mut indeterminate = 0;
    let mut not_encountered = 0;
    let mut pending = 0;
    for finding in findings {
        match finding.outcome {
            RuleComplianceOutcome::Compliant => compliant += 1,
            RuleComplianceOutcome::Violation => violation += 1,
            RuleComplianceOutcome::Indeterminate => indeterminate += 1,
            RuleComplianceOutcome::NotEncountered => not_encountered += 1,
            RuleComplianceOutcome::Pending => pending += 1,
        }
    }
    (compliant, violation, indeterminate, not_encountered, pending)
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(status: TrafficRuleEvidenceStatus) -> TrafficRuleEvidence {
        TrafficRuleEvidence {
            evidence_id: "rule-sign-0001".into(),
            source_id: "fixture-source-0001".into(),
            source_artifact_digest: "a".repeat(64),
            status,
        }
    }

    fn sample(timestamp_s: f64, progress_m: f64, speed_mps: f64, heading_rad: f64) -> RuleTrajectorySample {
        RuleTrajectorySample {
            timestamp_s,
            route_progress_m: progress_m,
            speed_mps,
            heading_rad,
        }
    }

    #[test]
    fn speed_limit_is_evaluated_only_when_rule_evidence_is_uncontradicted() {
        let rule = TrafficRuleRequirement {
            rule_id: "speed-30".into(),
            active_start_progress_m: 1.0,
            active_end_progress_m: 2.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 13.4,
                tolerance_mps: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![sample(0.0, 1.0, 10.0, 0.0), sample(0.1, 2.0, 14.0, 0.0)];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Violation);
        assert_eq!(
            report.findings[0].reason,
            RuleComplianceReason::SpeedLimitExceeded
        );
        assert!(report.identity_is_valid());
    }

    #[test]
    fn conflicted_or_unavailable_rule_evidence_is_indeterminate() {
        for status in [
            TrafficRuleEvidenceStatus::Conflicted,
            TrafficRuleEvidenceStatus::Unavailable,
        ] {
            let mut supplied_evidence = evidence(status);
            if status == TrafficRuleEvidenceStatus::Unavailable {
                supplied_evidence.source_id.clear();
                supplied_evidence.source_artifact_digest.clear();
            }
            let rule = TrafficRuleRequirement {
                rule_id: format!("speed-{status:?}"),
                active_start_progress_m: 0.0,
                active_end_progress_m: 100.0,
                kind: TrafficRuleKind::SpeedLimit {
                    maximum_speed_mps: 10.0,
                    tolerance_mps: 0.0,
                    maximum_sample_gap_s: 0.25,
                },
                evidence: supplied_evidence,
            };
            let report = evaluate_traffic_rules(
                &[rule],
                &[sample(0.0, 1.0, 40.0, 0.0), sample(0.1, 2.0, 40.0, 0.0)],
            )
            .unwrap();
            assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        }
    }

    #[test]
    fn sparse_samples_inside_speed_zone_are_indeterminate() {
        let rule = TrafficRuleRequirement {
            rule_id: "speed-sparse".into(),
            active_start_progress_m: 1.0,
            active_end_progress_m: 2.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 13.4,
                tolerance_mps: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 1.0, 10.0, 0.0),
            sample(1.0, 2.0, 10.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(report.findings[0].reason, RuleComplianceReason::TrajectorySamplingGap);
    }

    #[test]
    fn partial_rule_zone_coverage_is_indeterminate_not_compliant() {
        let rule = TrafficRuleRequirement {
            rule_id: "speed-partial-zone".into(),
            active_start_progress_m: 1.0,
            active_end_progress_m: 3.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 13.4,
                tolerance_mps: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 1.0, 10.0, 0.0),
            sample(0.1, 1.5, 10.0, 0.0),
            sample(0.2, 2.0, 10.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(
            report.findings[0].reason,
            RuleComplianceReason::TrajectoryCoverageInsufficient
        );
    }

    #[test]
    fn rule_zone_skipped_between_sparse_samples_is_not_not_encountered() {
        let rule = TrafficRuleRequirement {
            rule_id: "speed-skipped-zone".into(),
            active_start_progress_m: 10.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 13.4,
                tolerance_mps: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 5.0, 10.0, 0.0),
            sample(0.1, 25.0, 10.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(report.findings[0].reason, RuleComplianceReason::TrajectoryCoverageInsufficient);
    }

    #[test]
    fn malformed_evidence_digest_is_indeterminate_not_compliant() {
        let mut invalid_evidence = evidence(TrafficRuleEvidenceStatus::Uncontradicted);
        invalid_evidence.source_artifact_digest = "not-a-digest".into();
        let rule = TrafficRuleRequirement {
            rule_id: "speed-30".into(),
            active_start_progress_m: 0.0,
            active_end_progress_m: 100.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 13.4,
                tolerance_mps: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: invalid_evidence,
        };
        let report = evaluate_traffic_rules(
            &[rule],
            &[sample(0.0, 1.0, 10.0, 0.0), sample(0.1, 2.0, 10.0, 0.0)],
        )
        .unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(
            report.findings[0].reason,
            RuleComplianceReason::EvidenceMalformed
        );
    }

    #[test]
    fn mandatory_direction_obeys_angle_wraparound() {
        let rule = TrafficRuleRequirement {
            rule_id: "mandatory-left".into(),
            active_start_progress_m: 1.0,
            active_end_progress_m: 2.0,
            kind: TrafficRuleKind::MandatoryDirection {
                required_heading_rad: -std::f64::consts::PI + 0.05,
                maximum_heading_error_rad: 0.1,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 1.0, 2.0, std::f64::consts::PI - 0.05),
            sample(0.1, 2.0, 2.0, std::f64::consts::PI - 0.04),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Compliant);
    }

    #[test]
    fn mandatory_direction_sparse_samples_are_indeterminate() {
        let rule = TrafficRuleRequirement {
            rule_id: "mandatory-sparse".into(),
            active_start_progress_m: 1.0,
            active_end_progress_m: 2.0,
            kind: TrafficRuleKind::MandatoryDirection {
                required_heading_rad: 0.0,
                maximum_heading_error_rad: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 1.0, 2.0, 0.0),
            sample(1.0, 2.0, 2.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(report.findings[0].reason, RuleComplianceReason::TrajectorySamplingGap);
    }

    #[test]
    fn mandatory_direction_violation_is_located_in_trace() {
        let rule = TrafficRuleRequirement {
            rule_id: "mandatory-left".into(),
            active_start_progress_m: 1.0,
            active_end_progress_m: 2.0,
            kind: TrafficRuleKind::MandatoryDirection {
                required_heading_rad: 1.0,
                maximum_heading_error_rad: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![sample(0.0, 1.0, 2.0, 1.0), sample(0.1, 2.0, 2.0, 0.0)];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Violation);
        assert_eq!(report.findings[0].first_violating_sample_index, Some(1));
    }

    #[test]
    fn stop_line_requires_contiguous_stationary_dwell_before_crossing() {
        let rule = TrafficRuleRequirement {
            rule_id: "stop-line-001".into(),
            active_start_progress_m: 8.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::StopBeforeLine {
                stop_line_progress_m: 10.0,
                stop_speed_threshold_mps: 0.1,
                required_stationary_duration_s: 1.0,
                maximum_sample_gap_s: 0.6,
                progress_tolerance_m: 0.05,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let compliant = vec![
            sample(0.0, 8.0, 2.0, 0.0),
            sample(0.5, 9.0, 0.0, 0.0),
            sample(1.0, 9.0, 0.0, 0.0),
            sample(1.5, 9.0, 0.0, 0.0),
            sample(1.7, 10.2, 1.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule.clone()], &compliant).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Compliant);

        let noncompliant = vec![
            sample(0.0, 8.0, 2.0, 0.0),
            sample(0.5, 9.0, 0.5, 0.0),
            sample(1.0, 10.2, 2.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &noncompliant).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Violation);
        assert_eq!(report.findings[0].first_violating_sample_index, Some(2));
    }

    #[test]
    fn stop_line_zone_skipped_by_trajectory_is_indeterminate() {
        let rule = TrafficRuleRequirement {
            rule_id: "stop-line-skipped-zone".into(),
            active_start_progress_m: 8.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::StopBeforeLine {
                stop_line_progress_m: 10.0,
                stop_speed_threshold_mps: 0.1,
                required_stationary_duration_s: 1.0,
                maximum_sample_gap_s: 0.6,
                progress_tolerance_m: 0.05,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 7.0, 2.0, 0.0),
            sample(0.1, 11.0, 2.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(
            report.findings[0].reason,
            RuleComplianceReason::TrajectoryCoverageInsufficient
        );
    }

    #[test]
    fn stop_line_large_sample_gap_does_not_claim_continuous_dwell() {
        let rule = TrafficRuleRequirement {
            rule_id: "stop-line-gap".into(),
            active_start_progress_m: 8.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::StopBeforeLine {
                stop_line_progress_m: 10.0,
                stop_speed_threshold_mps: 0.1,
                required_stationary_duration_s: 1.0,
                maximum_sample_gap_s: 0.25,
                progress_tolerance_m: 0.05,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 8.0, 2.0, 0.0),
            sample(0.5, 9.0, 0.0, 0.0),
            sample(2.0, 9.0, 0.0, 0.0),
            sample(2.1, 10.2, 1.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(
            report.findings[0].reason,
            RuleComplianceReason::TrajectorySamplingGap
        );
    }

    #[test]
    fn post_stop_acceleration_gap_is_not_counted_as_stationary_dwell() {
        let rule = TrafficRuleRequirement {
            rule_id: "stop-line-acceleration-gap".into(),
            active_start_progress_m: 8.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::StopBeforeLine {
                stop_line_progress_m: 10.0,
                stop_speed_threshold_mps: 0.1,
                required_stationary_duration_s: 1.0,
                maximum_sample_gap_s: 1.5,
                progress_tolerance_m: 0.05,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 8.0, 2.0, 0.0),
            sample(0.5, 9.0, 0.0, 0.0),
            sample(1.0, 9.0, 0.0, 0.0),
            sample(2.0, 10.2, 2.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Violation);
    }

    #[test]
    fn stop_line_stationary_speed_with_impossible_progress_motion_is_rejected() {
        let rule = TrafficRuleRequirement {
            rule_id: "stop-line-motion".into(),
            active_start_progress_m: 8.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::StopBeforeLine {
                stop_line_progress_m: 10.0,
                stop_speed_threshold_mps: 0.1,
                required_stationary_duration_s: 1.0,
                maximum_sample_gap_s: 0.6,
                progress_tolerance_m: 0.02,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 8.0, 2.0, 0.0),
            sample(0.5, 8.8, 0.0, 0.0),
            sample(1.0, 9.6, 0.0, 0.0),
            sample(1.5, 10.2, 1.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(
            report.findings[0].reason,
            RuleComplianceReason::TrajectoryKinematicInconsistency
        );
    }

    #[test]
    fn stop_line_not_encountered_is_not_misreported_as_compliant() {
        let rule = TrafficRuleRequirement {
            rule_id: "stop-line-002".into(),
            active_start_progress_m: 7.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::StopBeforeLine {
                stop_line_progress_m: 10.0,
                stop_speed_threshold_mps: 0.1,
                required_stationary_duration_s: 1.0,
                maximum_sample_gap_s: 0.6,
                progress_tolerance_m: 0.05,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let report = evaluate_traffic_rules(
            &[rule],
            &[sample(0.0, 1.0, 3.0, 0.0), sample(0.5, 5.0, 3.0, 0.0)],
        )
        .unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::NotEncountered);
    }

    #[test]
    fn stop_line_starting_after_control_zone_entry_is_indeterminate() {
        let rule = TrafficRuleRequirement {
            rule_id: "stop-line-late-start".into(),
            active_start_progress_m: 5.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::StopBeforeLine {
                stop_line_progress_m: 10.0,
                stop_speed_threshold_mps: 0.1,
                required_stationary_duration_s: 1.0,
                maximum_sample_gap_s: 0.6,
                progress_tolerance_m: 0.05,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let trajectory = vec![
            sample(0.0, 8.0, 2.0, 0.0),
            sample(0.5, 9.0, 0.0, 0.0),
            sample(1.0, 9.0, 0.0, 0.0),
            sample(1.5, 10.2, 1.0, 0.0),
        ];
        let report = evaluate_traffic_rules(&[rule], &trajectory).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(
            report.findings[0].reason,
            RuleComplianceReason::TrajectoryCoverageInsufficient
        );
    }

    #[test]
    fn stop_line_episode_ending_inside_control_zone_is_pending() {
        let rule = TrafficRuleRequirement {
            rule_id: "stop-line-pending".into(),
            active_start_progress_m: 8.0,
            active_end_progress_m: 20.0,
            kind: TrafficRuleKind::StopBeforeLine {
                stop_line_progress_m: 10.0,
                stop_speed_threshold_mps: 0.1,
                required_stationary_duration_s: 1.0,
                maximum_sample_gap_s: 0.6,
                progress_tolerance_m: 0.05,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let report = evaluate_traffic_rules(
            &[rule],
            &[sample(0.0, 8.0, 2.0, 0.0), sample(0.5, 9.0, 0.0, 0.0)],
        )
        .unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Pending);
        assert_eq!(report.pending_count, 1);
        assert!(report.identity_is_valid());
    }

    #[test]
    fn nonmonotonic_trajectory_is_indeterminate_and_replayable() {
        let rule = TrafficRuleRequirement {
            rule_id: "speed-30".into(),
            active_start_progress_m: 0.0,
            active_end_progress_m: 100.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 13.4,
                tolerance_mps: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let invalid = vec![sample(0.0, 1.0, 10.0, 0.0), sample(0.0, 2.0, 10.0, 0.0)];
        let report = evaluate_traffic_rules(&[rule], &invalid).unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert!(report.identity_is_valid());
    }

    #[test]
    fn nonfinite_trajectory_and_rule_parameters_are_rejected_explicitly() {
        let rule = TrafficRuleRequirement {
            rule_id: "speed-30".into(),
            active_start_progress_m: 0.0,
            active_end_progress_m: 100.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 13.4,
                tolerance_mps: 0.2,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        assert_eq!(
            evaluate_traffic_rules(
                &[rule.clone()],
                &[sample(0.0, 1.0, f64::NAN, 0.0)]
            )
            .unwrap_err(),
            RuleEvaluationError::NonFiniteInput
        );
        let mut malformed_rule = rule;
        malformed_rule.active_end_progress_m = f64::INFINITY;
        assert_eq!(
            evaluate_traffic_rules(
                &[malformed_rule],
                &[sample(0.0, 1.0, 1.0, 0.0)]
            )
            .unwrap_err(),
            RuleEvaluationError::NonFiniteInput
        );
    }

    #[test]
    fn duplicate_rule_ids_and_empty_trajectory_are_rejected() {
        let rule = TrafficRuleRequirement {
            rule_id: "same-rule".into(),
            active_start_progress_m: 0.0,
            active_end_progress_m: 10.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 10.0,
                tolerance_mps: 0.0,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        assert_eq!(
            evaluate_traffic_rules(&[rule.clone()], &[]).unwrap_err(),
            RuleEvaluationError::EmptyTrajectory
        );
        assert_eq!(
            evaluate_traffic_rules(&[rule.clone(), rule], &[sample(0.0, 0.0, 1.0, 0.0)])
                .unwrap_err(),
            RuleEvaluationError::DuplicateRuleId
        );
        let mut empty_id = TrafficRuleRequirement {
            rule_id: String::new(),
            active_start_progress_m: 0.0,
            active_end_progress_m: 10.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 10.0,
                tolerance_mps: 0.0,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        empty_id.rule_id.clear();
        assert_eq!(
            evaluate_traffic_rules(
                &[empty_id],
                &[sample(0.0, 0.0, 1.0, 0.0)]
            )
            .unwrap_err(),
            RuleEvaluationError::EmptyRuleId
        );
    }

    #[test]
    fn unsupported_rule_families_are_retained_as_indeterminate() {
        let rule = TrafficRuleRequirement {
            rule_id: "yield-unknown-semantics".into(),
            active_start_progress_m: 0.0,
            active_end_progress_m: 30.0,
            kind: TrafficRuleKind::Unsupported {
                rule_family: "yield-right-of-way".into(),
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let report = evaluate_traffic_rules(
            &[rule],
            &[sample(0.0, 1.0, 8.0, 0.0), sample(0.1, 2.0, 8.0, 0.0)],
        )
        .unwrap();
        assert_eq!(report.findings[0].outcome, RuleComplianceOutcome::Indeterminate);
        assert_eq!(
            report.findings[0].reason,
            RuleComplianceReason::RuleKindUnsupported
        );
        assert!(report.identity_is_valid());
    }

    #[test]
    fn report_recomputes_findings_and_detects_digest_recalculation_tampering() {
        let rule = TrafficRuleRequirement {
            rule_id: "speed-30".into(),
            active_start_progress_m: 1.0,
            active_end_progress_m: 2.0,
            kind: TrafficRuleKind::SpeedLimit {
                maximum_speed_mps: 10.0,
                tolerance_mps: 0.0,
                maximum_sample_gap_s: 0.25,
            },
            evidence: evidence(TrafficRuleEvidenceStatus::Uncontradicted),
        };
        let mut report = evaluate_traffic_rules(
            &[rule],
            &[sample(0.0, 1.0, 2.0, 0.0), sample(0.1, 2.0, 2.0, 0.0)],
        )
        .unwrap();
        assert!(report.identity_is_valid());
        report.findings[0].outcome = RuleComplianceOutcome::Violation;
        report.report_digest.clear();
        let bytes = serde_json::to_vec(&report).unwrap();
        report.report_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!report.identity_is_valid());
    }
}
