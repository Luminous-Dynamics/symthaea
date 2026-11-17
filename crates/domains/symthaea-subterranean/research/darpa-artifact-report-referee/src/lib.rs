//! Independent one-to-one evaluator for bounded artifact reports.
//!
//! Ground-truth records are evaluator-only. Candidate generation must not
//! receive this catalogue, its coordinates, its type labels or derived truth.
//! This package is a synthetic/reference scoring kernel, not a perception,
//! SLAM, mission-control, safety or authorization component.

use std::collections::{BTreeMap, BTreeSet};

const MAX_GROUND_TRUTH: usize = 4_096;
const MAX_CANDIDATE_REPORTS: usize = 4_096;
const MAX_IDENTIFIER_BYTES: usize = 256;
const MAX_TYPE_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn validate(self) -> Result<(), RefereeError> {
        if [self.x, self.y, self.z]
            .into_iter()
            .all(f64::is_finite)
        {
            Ok(())
        } else {
            Err(RefereeError::NonFiniteCoordinate)
        }
    }

    fn distance(self, other: Self) -> Result<f64, RefereeError> {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        if !dx.is_finite() || !dy.is_finite() || !dz.is_finite() {
            return Err(RefereeError::NumericalOverflow);
        }
        let distance = dx.hypot(dy).hypot(dz);
        if distance.is_finite() {
            Ok(distance)
        } else {
            Err(RefereeError::NumericalOverflow)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroundTruthArtifact {
    pub id: String,
    /// Original source type label; never normalized in this crate.
    pub source_type: String,
    pub position: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateReport {
    pub id: String,
    /// Original report type label; exact string match is required by v0.1.
    pub source_type: String,
    pub position: Vec3,
    /// Higher values are considered first when a report budget is imposed.
    pub priority: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReportProfile {
    /// Maximum number of candidate reports admitted to scoring. None means
    /// that the evaluator applies no report cap.
    pub maximum_reports: Option<usize>,
    /// Explicit Euclidean 3-D localization tolerance in metres.
    pub localization_tolerance_m: f64,
}

impl ReportProfile {
    pub fn validate(self) -> Result<Self, RefereeError> {
        if !self.localization_tolerance_m.is_finite()
            || self.localization_tolerance_m < 0.0
        {
            return Err(RefereeError::InvalidLocalizationTolerance);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactMatch {
    pub report_id: String,
    pub ground_truth_id: String,
    pub source_type: String,
    pub localization_error_m: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerClassMetrics {
    pub source_type: String,
    pub ground_truth_count: usize,
    pub selected_report_count: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationReport {
    pub ground_truth_count: usize,
    pub submitted_report_count: usize,
    pub selected_report_count: usize,
    pub budget_excluded_report_count: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    /// None when no reports were selected.
    pub precision: Option<f64>,
    /// None only when the ground-truth catalogue is empty (rejected in v0.1).
    pub recall: Option<f64>,
    pub localization_tolerance_m: f64,
    pub matches: Vec<ArtifactMatch>,
    pub unmatched_selected_report_ids: Vec<String>,
    pub budget_excluded_report_ids: Vec<String>,
    pub per_class: Vec<PerClassMetrics>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefereeError {
    EmptyGroundTruth,
    TooManyGroundTruth,
    TooManyCandidateReports,
    EmptyIdentifier(&'static str),
    IdentifierTooLong(&'static str),
    DuplicateGroundTruthId(String),
    DuplicateCandidateReportId(String),
    EmptyTypeLabel,
    TypeLabelTooLong,
    NonFiniteCoordinate,
    NumericalOverflow,
    InvalidLocalizationTolerance,
}

impl std::fmt::Display for RefereeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RefereeError {}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), RefereeError> {
    if value.trim().is_empty() {
        return Err(RefereeError::EmptyIdentifier(field));
    }
    if value.len() > MAX_IDENTIFIER_BYTES {
        return Err(RefereeError::IdentifierTooLong(field));
    }
    Ok(())
}

fn validate_type_label(value: &str) -> Result<(), RefereeError> {
    if value.trim().is_empty() {
        return Err(RefereeError::EmptyTypeLabel);
    }
    if value.len() > MAX_TYPE_BYTES {
        return Err(RefereeError::TypeLabelTooLong);
    }
    Ok(())
}

fn validate_inputs(
    ground_truth: &[GroundTruthArtifact],
    reports: &[CandidateReport],
    profile: ReportProfile,
) -> Result<(), RefereeError> {
    profile.validate()?;
    if ground_truth.is_empty() {
        return Err(RefereeError::EmptyGroundTruth);
    }
    if ground_truth.len() > MAX_GROUND_TRUTH {
        return Err(RefereeError::TooManyGroundTruth);
    }
    if reports.len() > MAX_CANDIDATE_REPORTS {
        return Err(RefereeError::TooManyCandidateReports);
    }

    let mut truth_ids = BTreeSet::new();
    for artifact in ground_truth {
        validate_identifier(&artifact.id, "ground_truth.id")?;
        validate_type_label(&artifact.source_type)?;
        artifact.position.validate()?;
        if !truth_ids.insert(artifact.id.as_str()) {
            return Err(RefereeError::DuplicateGroundTruthId(artifact.id.clone()));
        }
    }

    let mut report_ids = BTreeSet::new();
    for report in reports {
        validate_identifier(&report.id, "candidate_report.id")?;
        validate_type_label(&report.source_type)?;
        report.position.validate()?;
        if !report_ids.insert(report.id.as_str()) {
            return Err(RefereeError::DuplicateCandidateReportId(
                report.id.clone(),
            ));
        }
    }
    Ok(())
}

/// Select candidate reports under the declared budget. This operation is
/// deterministic and does not inspect ground truth.
fn select_reports<'a>(
    reports: &'a [CandidateReport],
    maximum_reports: Option<usize>,
) -> (Vec<&'a CandidateReport>, Vec<String>) {
    let mut ordered = reports.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| left.id.cmp(&right.id))
    });

    let selected_count = maximum_reports.unwrap_or(ordered.len()).min(ordered.len());
    let excluded = ordered
        .iter()
        .skip(selected_count)
        .map(|report| report.id.clone())
        .collect::<Vec<_>>();
    ordered.truncate(selected_count);
    (ordered, excluded)
}

/// Find a maximum-cardinality one-to-one assignment using deterministic
/// augmenting paths. Adjacency entries are already ordered by localization
/// error then ground-truth ID. This maximizes accepted report count; it does
/// not claim to minimize the total localization error across all assignments.
fn try_assign(
    report_index: usize,
    adjacency: &[Vec<(usize, f64)>],
    visited_truth: &mut [bool],
    owner_by_truth: &mut [Option<usize>],
) -> bool {
    for (truth_index, _) in &adjacency[report_index] {
        if visited_truth[*truth_index] {
            continue;
        }
        visited_truth[*truth_index] = true;

        let can_take = match owner_by_truth[*truth_index] {
            None => true,
            Some(previous_report) => {
                try_assign(previous_report, adjacency, visited_truth, owner_by_truth)
            }
        };
        if can_take {
            owner_by_truth[*truth_index] = Some(report_index);
            return true;
        }
    }
    false
}

/// Evaluate reports against ground truth using exact type-label equality and
/// a declared Euclidean 3-D distance threshold.
///
/// Matching is one-to-one: duplicate reports cannot claim the same artifact
/// twice. Reports over the optional budget are excluded before matching.
pub fn evaluate_reports(
    ground_truth: &[GroundTruthArtifact],
    reports: &[CandidateReport],
    profile: ReportProfile,
) -> Result<EvaluationReport, RefereeError> {
    validate_inputs(ground_truth, reports, profile)?;
    let (selected, budget_excluded_report_ids) =
        select_reports(reports, profile.maximum_reports);

    // Sort source records by stable ID so input-order permutation does not
    // change adjacency, assignments or normalized reports.
    let mut ordered_truth = ground_truth.iter().collect::<Vec<_>>();
    ordered_truth.sort_by(|a, b| a.id.cmp(&b.id));

    let mut adjacency = vec![Vec::<(usize, f64)>::new(); selected.len()];
    for (report_index, report) in selected.iter().enumerate() {
        for (truth_index, truth) in ordered_truth.iter().enumerate() {
            if report.source_type != truth.source_type {
                continue;
            }
            let error = report.position.distance(truth.position)?;
            if error <= profile.localization_tolerance_m {
                adjacency[report_index].push((truth_index, error));
            }
        }
        adjacency[report_index].sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| ordered_truth[left.0].id.cmp(&ordered_truth[right.0].id))
        });
    }

    let mut owner_by_truth = vec![None; ordered_truth.len()];
    for report_index in 0..selected.len() {
        let mut visited_truth = vec![false; ordered_truth.len()];
        let _ = try_assign(
            report_index,
            &adjacency,
            &mut visited_truth,
            &mut owner_by_truth,
        );
    }

    let mut matches = Vec::new();
    let mut matched_report_ids = BTreeSet::new();
    for (truth_index, report_owner) in owner_by_truth.iter().enumerate() {
        let Some(report_index) = report_owner else {
            continue;
        };
        let report = selected[*report_index];
        let truth = ordered_truth[truth_index];
        let error = report.position.distance(truth.position)?;
        matches.push(ArtifactMatch {
            report_id: report.id.clone(),
            ground_truth_id: truth.id.clone(),
            source_type: truth.source_type.clone(),
            localization_error_m: error,
        });
        matched_report_ids.insert(report.id.as_str());
    }
    matches.sort_by(|a, b| {
        a.report_id
            .cmp(&b.report_id)
            .then_with(|| a.ground_truth_id.cmp(&b.ground_truth_id))
    });

    let unmatched_selected_report_ids = selected
        .iter()
        .filter(|report| !matched_report_ids.contains(report.id.as_str()))
        .map(|report| report.id.clone())
        .collect::<Vec<_>>();

    let true_positive = matches.len();
    let selected_report_count = selected.len();
    let ground_truth_count = ground_truth.len();
    let false_positive = selected_report_count - true_positive;
    let false_negative = ground_truth_count - true_positive;
    let precision = if selected_report_count == 0 {
        None
    } else {
        Some(true_positive as f64 / selected_report_count as f64)
    };
    let recall = Some(true_positive as f64 / ground_truth_count as f64);

    let mut class_counts = BTreeMap::<String, (usize, usize, usize)>::new();
    for truth in &ordered_truth {
        class_counts
            .entry(truth.source_type.clone())
            .or_default()
            .0 += 1;
    }
    for report in &selected {
        class_counts
            .entry(report.source_type.clone())
            .or_default()
            .1 += 1;
    }
    for matched in &matches {
        if let Some(counts) = class_counts.get_mut(&matched.source_type) {
            counts.2 += 1;
        }
    }
    let per_class = class_counts
        .into_iter()
        .map(|(source_type, (ground_truth_count, selected_report_count, true_positive))| {
            PerClassMetrics {
                source_type,
                ground_truth_count,
                selected_report_count,
                true_positive,
                false_positive: selected_report_count - true_positive,
                false_negative: ground_truth_count - true_positive,
            }
        })
        .collect();

    Ok(EvaluationReport {
        ground_truth_count,
        submitted_report_count: reports.len(),
        selected_report_count,
        budget_excluded_report_count: budget_excluded_report_ids.len(),
        true_positive,
        false_positive,
        false_negative,
        precision,
        recall,
        localization_tolerance_m: profile.localization_tolerance_m,
        matches,
        unmatched_selected_report_ids,
        budget_excluded_report_ids,
        per_class,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth(id: &str, class: &str, x: f64, y: f64, z: f64) -> GroundTruthArtifact {
        GroundTruthArtifact {
            id: id.into(),
            source_type: class.into(),
            position: Vec3::new(x, y, z),
        }
    }

    fn report(
        id: &str,
        class: &str,
        x: f64,
        y: f64,
        z: f64,
        priority: i32,
    ) -> CandidateReport {
        CandidateReport {
            id: id.into(),
            source_type: class.into(),
            position: Vec3::new(x, y, z),
            priority,
        }
    }

    fn profile(tolerance: f64, maximum_reports: Option<usize>) -> ReportProfile {
        ReportProfile {
            maximum_reports,
            localization_tolerance_m: tolerance,
        }
    }

    #[test]
    fn exact_type_and_three_dimensional_distance_are_required() {
        let gt = vec![truth("g1", "cellphone", 0.0, 0.0, 0.0)];
        let reports = vec![
            report("r1", "cellphone", 3.0, 4.0, 0.0, 2),
            report("r2", "cell_phone", 0.0, 0.0, 0.0, 1),
        ];
        let result = evaluate_reports(&gt, &reports, profile(5.0, None)).unwrap();
        assert_eq!(result.true_positive, 1);
        assert_eq!(result.false_positive, 1);
        assert_eq!(result.false_negative, 0);
        assert_eq!(result.matches[0].report_id, "r1");
        assert!((result.matches[0].localization_error_m - 5.0).abs() < 1.0e-12);
        assert_eq!(result.per_class.len(), 2);
    }

    #[test]
    fn vertical_error_contributes_to_euclidean_distance() {
        let gt = vec![truth("g1", "vent", 0.0, 0.0, 0.0)];
        let reports = vec![report("r1", "vent", 0.0, 0.0, 5.01, 1)];
        let result = evaluate_reports(&gt, &reports, profile(5.0, None)).unwrap();
        assert_eq!(result.true_positive, 0);
        assert_eq!(result.false_positive, 1);
        assert_eq!(result.false_negative, 1);
    }

    #[test]
    fn duplicate_reports_cannot_match_one_ground_truth_twice() {
        let gt = vec![truth("g1", "survivor", 1.0, 2.0, 3.0)];
        let reports = vec![
            report("r1", "survivor", 1.0, 2.0, 3.0, 2),
            report("r2", "survivor", 1.1, 2.0, 3.0, 1),
        ];
        let result = evaluate_reports(&gt, &reports, profile(5.0, None)).unwrap();
        assert_eq!(result.true_positive, 1);
        assert_eq!(result.false_positive, 1);
        assert_eq!(result.false_negative, 0);
    }

    #[test]
    fn maximum_cardinality_matching_avoids_greedy_nearest_report_trap() {
        // r1 can match either truth; r2 can only match g1. An irreversible
        // nearest-first assignment of r1->g1 would lose a valid second match.
        let gt = vec![
            truth("g1", "vent", 0.0, 0.0, 0.0),
            truth("g2", "vent", 4.0, 0.0, 0.0),
        ];
        let reports = vec![
            report("r1", "vent", 1.0, 0.0, 0.0, 2),
            report("r2", "vent", -1.0, 0.0, 0.0, 1),
        ];
        let result = evaluate_reports(&gt, &reports, profile(3.0, None)).unwrap();
        assert_eq!(result.true_positive, 2);
        assert_eq!(result.false_positive, 0);
        assert_eq!(result.false_negative, 0);
    }

    #[test]
    fn report_budget_selects_highest_priority_with_stable_tie_break() {
        let gt = vec![
            truth("g1", "vent", 0.0, 0.0, 0.0),
            truth("g2", "gas", 10.0, 0.0, 0.0),
        ];
        let reports = vec![
            report("r-low", "vent", 0.0, 0.0, 0.0, 1),
            report("r-b", "gas", 10.0, 0.0, 0.0, 3),
            report("r-a", "vent", 0.0, 0.0, 0.0, 3),
        ];
        let result = evaluate_reports(&gt, &reports, profile(1.0, Some(2))).unwrap();
        assert_eq!(result.selected_report_count, 2);
        assert_eq!(result.budget_excluded_report_ids, vec!["r-low"]);
        assert_eq!(result.true_positive, 2);
    }

    #[test]
    fn wrong_class_and_out_of_tolerance_reports_are_false_positive_and_negative() {
        let gt = vec![truth("g1", "gas", 0.0, 0.0, 0.0)];
        let reports = vec![
            report("wrong-type", "vent", 0.0, 0.0, 0.0, 2),
            report("far-away", "gas", 6.0, 0.0, 0.0, 1),
        ];
        let result = evaluate_reports(&gt, &reports, profile(5.0, None)).unwrap();
        assert_eq!(result.true_positive, 0);
        assert_eq!(result.false_positive, 2);
        assert_eq!(result.false_negative, 1);
    }

    #[test]
    fn result_is_invariant_to_input_record_order() {
        let gt_a = vec![
            truth("g2", "rope", 10.0, 0.0, 0.0),
            truth("g1", "vent", 0.0, 0.0, 0.0),
        ];
        let reports_a = vec![
            report("r2", "rope", 10.1, 0.0, 0.0, 1),
            report("r1", "vent", 0.0, 0.0, 0.0, 2),
        ];
        let mut gt_b = gt_a.clone();
        gt_b.reverse();
        let mut reports_b = reports_a.clone();
        reports_b.reverse();
        let left = evaluate_reports(&gt_a, &reports_a, profile(1.0, None)).unwrap();
        let right = evaluate_reports(&gt_b, &reports_b, profile(1.0, None)).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn malformed_or_duplicate_records_fail_closed() {
        let mut gt = vec![truth("g1", "vent", 0.0, 0.0, 0.0)];
        gt.push(truth("g1", "gas", 1.0, 0.0, 0.0));
        assert!(matches!(
            evaluate_reports(&gt, &[], profile(5.0, None)),
            Err(RefereeError::DuplicateGroundTruthId(_))
        ));

        let mut reports = vec![report("r1", "vent", 0.0, 0.0, 0.0, 1)];
        reports.push(report("r1", "vent", 0.1, 0.0, 0.0, 1));
        assert!(matches!(
            evaluate_reports(&[truth("g1", "vent", 0.0, 0.0, 0.0)], &reports, profile(5.0, None)),
            Err(RefereeError::DuplicateCandidateReportId(_))
        ));
    }

    #[test]
    fn non_finite_coordinates_and_invalid_tolerances_are_rejected() {
        let gt = vec![truth("g1", "vent", 0.0, 0.0, 0.0)];
        let reports = vec![report("r1", "vent", f64::NAN, 0.0, 0.0, 1)];
        assert_eq!(
            evaluate_reports(&gt, &reports, profile(5.0, None)),
            Err(RefereeError::NonFiniteCoordinate)
        );
        assert_eq!(
            evaluate_reports(&gt, &[], profile(f64::NAN, None)),
            Err(RefereeError::InvalidLocalizationTolerance)
        );
    }

    #[test]
    fn empty_selected_reports_produce_unavailable_precision_and_zero_recall() {
        let gt = vec![truth("g1", "helmet", 0.0, 0.0, 0.0)];
        let result = evaluate_reports(&gt, &[], profile(5.0, Some(0))).unwrap();
        assert_eq!(result.precision, None);
        assert_eq!(result.recall, Some(0.0));
        assert_eq!(result.false_negative, 1);
    }
}
