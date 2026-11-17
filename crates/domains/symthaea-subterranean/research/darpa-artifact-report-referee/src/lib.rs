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
        if [self.x, self.y, self.z].into_iter().all(f64::is_finite) {
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
    /// Exact coordinate-frame identity of the source position.
    pub coordinate_frame_id: String,
    /// Original source type label; never normalized in this crate.
    pub source_type: String,
    pub position: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateReport {
    pub id: String,
    /// Coordinate frame of the report; must exactly match the profile frame.
    pub coordinate_frame_id: String,
    /// Original report type label; exact string match is required by v0.1.
    pub source_type: String,
    pub position: Vec3,
    /// Monotonic submission order; lower values are received first.
    pub submission_sequence: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReportProfile {
    /// All input coordinates must already be expressed in this exact frame.
    /// The evaluator does not apply or infer coordinate transformations.
    pub coordinate_frame_id: String,
    /// Maximum number of candidate reports admitted to scoring. None means
    /// that the evaluator applies no report cap.
    pub maximum_reports: Option<usize>,
    /// Explicit Euclidean 3-D localization tolerance in metres.
    pub localization_tolerance_m: f64,
}

impl ReportProfile {
    pub fn validate(&self) -> Result<(), RefereeError> {
        validate_identifier(&self.coordinate_frame_id, "profile.coordinate_frame_id")?;
        if !self.localization_tolerance_m.is_finite() || self.localization_tolerance_m < 0.0 {
            return Err(RefereeError::InvalidLocalizationTolerance);
        }
        Ok(())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemsFinalRound {
    Preliminary,
    Prize,
}

impl SystemsFinalRound {
    pub const fn expected_artifacts(self) -> usize {
        match self {
            Self::Preliminary => 20,
            Self::Prize => 40,
        }
    }

    pub const fn maximum_reports(self) -> usize {
        match self {
            Self::Preliminary => 25,
            Self::Prize => 45,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactTypeAlias {
    /// Raw type string used in the ground-truth CSV source.
    pub source_type: String,
    /// Exact report string in the official DARPA Finals artifacts specification.
    pub official_report_string: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactTypeCrosswalk {
    /// Version/date of the externally reviewed source-to-official mapping.
    pub version: String,
    pub mappings: Vec<ArtifactTypeAlias>,
}

/// Crosswalk for the DARPA SubT Finals Artifacts Specification dated
/// 2021-04-01. Keep this mapping explicit; do not lowercase, trim or infer
/// class aliases during scoring.
pub fn darpa_final_2021_crosswalk_v0_1() -> ArtifactTypeCrosswalk {
    ArtifactTypeCrosswalk {
        version: "darpa-finals-artifact-strings-2021-04-01-v0.1".into(),
        mappings: vec![
            ArtifactTypeAlias {
                source_type: "survivor".into(),
                official_report_string: "Survivor".into(),
            },
            ArtifactTypeAlias {
                source_type: "cellphone".into(),
                official_report_string: "Cell Phone".into(),
            },
            ArtifactTypeAlias {
                source_type: "backpack".into(),
                official_report_string: "Backpack".into(),
            },
            ArtifactTypeAlias {
                source_type: "drill".into(),
                official_report_string: "Drill".into(),
            },
            ArtifactTypeAlias {
                source_type: "fireextinguisher".into(),
                official_report_string: "Fire Extinguisher".into(),
            },
            ArtifactTypeAlias {
                source_type: "gas".into(),
                official_report_string: "Gas".into(),
            },
            ArtifactTypeAlias {
                source_type: "vent".into(),
                official_report_string: "Vent".into(),
            },
            ArtifactTypeAlias {
                source_type: "helmet".into(),
                official_report_string: "Helmet".into(),
            },
            ArtifactTypeAlias {
                source_type: "rope".into(),
                official_report_string: "Rope".into(),
            },
            ArtifactTypeAlias {
                source_type: "cube".into(),
                official_report_string: "Cube".into(),
            },
        ],
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemsFinalProfile {
    pub round: SystemsFinalRound,
    pub coordinate_frame_id: String,
    pub type_crosswalk: ArtifactTypeCrosswalk,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SystemsFinalEvaluation {
    pub round: SystemsFinalRound,
    pub crosswalk_version: String,
    pub expected_artifact_count: usize,
    pub report_limit: usize,
    pub submitted_report_count: usize,
    pub processed_report_count: usize,
    pub score_points: usize,
    pub invalid_report_count: usize,
    pub false_negative_count: usize,
    pub precision: Option<f64>,
    pub recall: f64,
    pub matches: Vec<ArtifactMatch>,
    pub invalid_report_ids: Vec<String>,
    pub budget_rejected_report_ids: Vec<String>,
    pub unreported_ground_truth_ids: Vec<String>,
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
    DuplicateSubmissionSequence(u64),
    EmptyTypeLabel,
    TypeLabelTooLong,
    NonFiniteCoordinate,
    NumericalOverflow,
    InvalidLocalizationTolerance,
    CoordinateFrameMismatch {
        record_id: String,
        expected_frame: String,
        actual_frame: String,
    },
    UnexpectedGroundTruthCount {
        expected: usize,
        actual: usize,
    },
    EmptyTypeCrosswalk,
    DuplicateSourceTypeMapping(String),
    DuplicateOfficialTypeString(String),
    UnsupportedOfficialTypeString(String),
    MissingSourceTypeMapping(String),
}

impl std::fmt::Display for RefereeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RefereeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroundTruthParseError {
    EmptySourceFileId,
    EmptyCoordinateFrameId,
    InvalidFieldCount { line: usize, fields: usize },
    EmptyTypeLabel { line: usize },
    InvalidCoordinate { line: usize, column: usize },
    NonFiniteCoordinate { line: usize, column: usize },
}

/// Parse the source DARPA tab-separated artifact format:
/// `type<TAB>x<TAB>y<TAB>z`, optionally followed by one empty trailing
/// tab field. No header row is expected. The raw type string is preserved.
/// Record IDs are deterministic `source_file_id:line_number` values.
pub fn parse_ground_truth_tsv(
    source_file_id: &str,
    coordinate_frame_id: &str,
    input: &str,
) -> Result<Vec<GroundTruthArtifact>, GroundTruthParseError> {
    if source_file_id.trim().is_empty() {
        return Err(GroundTruthParseError::EmptySourceFileId);
    }
    if coordinate_frame_id.trim().is_empty() {
        return Err(GroundTruthParseError::EmptyCoordinateFrameId);
    }

    let mut artifacts = Vec::new();
    for (index, raw_line) in input.lines().enumerate() {
        let line_number = index + 1;
        if raw_line.trim().is_empty() {
            continue;
        }
        let mut fields = raw_line.split('\t').collect::<Vec<_>>();
        if fields.len() == 5 && fields[4].is_empty() {
            fields.pop();
        }
        if fields.len() != 4 {
            return Err(GroundTruthParseError::InvalidFieldCount {
                line: line_number,
                fields: fields.len(),
            });
        }
        if fields[0].trim().is_empty() {
            return Err(GroundTruthParseError::EmptyTypeLabel { line: line_number });
        }

        let mut coordinates = [0.0; 3];
        for (column, value) in fields[1..4].iter().enumerate() {
            let coordinate =
                value
                    .parse::<f64>()
                    .map_err(|_| GroundTruthParseError::InvalidCoordinate {
                        line: line_number,
                        column: column + 1,
                    })?;
            if !coordinate.is_finite() {
                return Err(GroundTruthParseError::NonFiniteCoordinate {
                    line: line_number,
                    column: column + 1,
                });
            }
            coordinates[column] = coordinate;
        }

        artifacts.push(GroundTruthArtifact {
            id: format!("{source_file_id}:{line_number}"),
            coordinate_frame_id: coordinate_frame_id.to_owned(),
            source_type: fields[0].to_owned(),
            position: Vec3::new(coordinates[0], coordinates[1], coordinates[2]),
        });
    }
    Ok(artifacts)
}

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
        validate_identifier(
            &artifact.coordinate_frame_id,
            "ground_truth.coordinate_frame_id",
        )?;
        if artifact.coordinate_frame_id != profile.coordinate_frame_id {
            return Err(RefereeError::CoordinateFrameMismatch {
                record_id: artifact.id.clone(),
                expected_frame: profile.coordinate_frame_id.clone(),
                actual_frame: artifact.coordinate_frame_id.clone(),
            });
        }
        validate_type_label(&artifact.source_type)?;
        artifact.position.validate()?;
        if !truth_ids.insert(artifact.id.as_str()) {
            return Err(RefereeError::DuplicateGroundTruthId(artifact.id.clone()));
        }
    }

    let mut report_ids = BTreeSet::new();
    let mut submission_sequences = BTreeSet::new();
    for report in reports {
        validate_identifier(&report.id, "candidate_report.id")?;
        if !submission_sequences.insert(report.submission_sequence) {
            return Err(RefereeError::DuplicateSubmissionSequence(
                report.submission_sequence,
            ));
        }
        validate_identifier(
            &report.coordinate_frame_id,
            "candidate_report.coordinate_frame_id",
        )?;
        if report.coordinate_frame_id != profile.coordinate_frame_id {
            return Err(RefereeError::CoordinateFrameMismatch {
                record_id: report.id.clone(),
                expected_frame: profile.coordinate_frame_id.clone(),
                actual_frame: report.coordinate_frame_id.clone(),
            });
        }
        validate_type_label(&report.source_type)?;
        report.position.validate()?;
        if !report_ids.insert(report.id.as_str()) {
            return Err(RefereeError::DuplicateCandidateReportId(report.id.clone()));
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
        left.submission_sequence
            .cmp(&right.submission_sequence)
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
    let (selected, budget_excluded_report_ids) = select_reports(reports, profile.maximum_reports);

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
        class_counts.entry(truth.source_type.clone()).or_default().0 += 1;
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
        .map(
            |(source_type, (ground_truth_count, selected_report_count, true_positive))| {
                PerClassMetrics {
                    source_type,
                    ground_truth_count,
                    selected_report_count,
                    true_positive,
                    false_positive: selected_report_count - true_positive,
                    false_negative: ground_truth_count - true_positive,
                }
            },
        )
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

fn validate_type_crosswalk(
    crosswalk: &ArtifactTypeCrosswalk,
    ground_truth: &[GroundTruthArtifact],
) -> Result<BTreeMap<String, String>, RefereeError> {
    validate_identifier(&crosswalk.version, "type_crosswalk.version")?;
    if crosswalk.mappings.is_empty() {
        return Err(RefereeError::EmptyTypeCrosswalk);
    }

    const OFFICIAL_TYPES: [&str; 10] = [
        "Survivor",
        "Cell Phone",
        "Backpack",
        "Drill",
        "Fire Extinguisher",
        "Gas",
        "Vent",
        "Helmet",
        "Rope",
        "Cube",
    ];

    let mut source_to_official = BTreeMap::new();
    let mut official_seen = BTreeSet::new();
    for mapping in &crosswalk.mappings {
        validate_type_label(&mapping.source_type)?;
        validate_type_label(&mapping.official_report_string)?;
        if !OFFICIAL_TYPES.contains(&mapping.official_report_string.as_str()) {
            return Err(RefereeError::UnsupportedOfficialTypeString(
                mapping.official_report_string.clone(),
            ));
        }
        if source_to_official
            .insert(
                mapping.source_type.clone(),
                mapping.official_report_string.clone(),
            )
            .is_some()
        {
            return Err(RefereeError::DuplicateSourceTypeMapping(
                mapping.source_type.clone(),
            ));
        }
        if !official_seen.insert(mapping.official_report_string.as_str()) {
            return Err(RefereeError::DuplicateOfficialTypeString(
                mapping.official_report_string.clone(),
            ));
        }
    }

    for artifact in ground_truth {
        if !source_to_official.contains_key(&artifact.source_type) {
            return Err(RefereeError::MissingSourceTypeMapping(
                artifact.source_type.clone(),
            ));
        }
    }
    Ok(source_to_official)
}

/// Replay the historical DARPA Systems Final artifact-report scoring
/// semantics in submission order.
///
/// Preliminary rounds expect 20 ground-truth artifacts and allow 25 reports;
/// the Prize round expects 40 artifacts and allows 45 reports. A report earns
/// one point only when its raw type label exactly matches an unreported
/// artifact and its Euclidean 3-D error is <= 5 m. Invalid reports may be
/// retried later; once an artifact has been reported successfully, subsequent
/// reports cannot score against it and still consume the report allowance.
///
/// The caller is responsible for applying any separately reviewed,
/// versioned source-label-to-official-label crosswalk before evaluation.
/// Timing tiebreakers and API/transport behavior are outside this v0.1 scorer.
pub fn evaluate_systems_final_stream(
    ground_truth: &[GroundTruthArtifact],
    reports: &[CandidateReport],
    profile: SystemsFinalProfile,
) -> Result<SystemsFinalEvaluation, RefereeError> {
    let expected_artifacts = profile.round.expected_artifacts();
    if ground_truth.len() != expected_artifacts {
        return Err(RefereeError::UnexpectedGroundTruthCount {
            expected: expected_artifacts,
            actual: ground_truth.len(),
        });
    }

    let source_to_official = validate_type_crosswalk(&profile.type_crosswalk, ground_truth)?;
    let report_limit = profile.round.maximum_reports();
    let scoring_profile = ReportProfile {
        coordinate_frame_id: profile.coordinate_frame_id.clone(),
        maximum_reports: Some(report_limit),
        localization_tolerance_m: 5.0,
    };
    validate_inputs(ground_truth, reports, scoring_profile)?;
    let (selected, budget_rejected_report_ids) = select_reports(reports, Some(report_limit));

    let mut ordered_truth = ground_truth.iter().collect::<Vec<_>>();
    ordered_truth.sort_by(|left, right| left.id.cmp(&right.id));
    let mut already_reported = vec![false; ordered_truth.len()];
    let mut matches = Vec::new();
    let mut invalid_report_ids = Vec::new();

    // This is intentionally a sequential state machine. An invalid report
    // leaves the truth item available for a later corrected report. A valid
    // report marks that truth item, so any later duplicate cannot score for it.
    for report in &selected {
        let mut best_match: Option<(usize, f64)> = None;
        for (truth_index, truth) in ordered_truth.iter().enumerate() {
            let official_type = source_to_official
                .get(&truth.source_type)
                .ok_or_else(|| RefereeError::MissingSourceTypeMapping(truth.source_type.clone()))?;
            if already_reported[truth_index] || report.source_type != *official_type {
                continue;
            }
            let error = report.position.distance(truth.position)?;
            if error > 5.0 {
                continue;
            }
            let should_replace = match best_match {
                None => true,
                Some((best_index, best_error)) => {
                    error < best_error
                        || (error == best_error && truth.id < ordered_truth[best_index].id)
                }
            };
            if should_replace {
                best_match = Some((truth_index, error));
            }
        }

        if let Some((truth_index, error)) = best_match {
            already_reported[truth_index] = true;
            let truth = ordered_truth[truth_index];
            matches.push(ArtifactMatch {
                report_id: report.id.clone(),
                ground_truth_id: truth.id.clone(),
                source_type: truth.source_type.clone(),
                localization_error_m: error,
            });
        } else {
            invalid_report_ids.push(report.id.clone());
        }
    }

    let mut unreported_ground_truth_ids = ordered_truth
        .iter()
        .enumerate()
        .filter(|(index, _)| !already_reported[*index])
        .map(|(_, artifact)| artifact.id.clone())
        .collect::<Vec<_>>();
    unreported_ground_truth_ids.sort();

    let score_points = matches.len();
    let processed_report_count = selected.len();
    let invalid_report_count = invalid_report_ids.len();
    let false_negative_count = unreported_ground_truth_ids.len();
    let precision = if processed_report_count == 0 {
        None
    } else {
        Some(score_points as f64 / processed_report_count as f64)
    };
    let recall = score_points as f64 / expected_artifacts as f64;

    Ok(SystemsFinalEvaluation {
        round: profile.round,
        crosswalk_version: profile.type_crosswalk.version,
        expected_artifact_count: expected_artifacts,
        report_limit,
        submitted_report_count: reports.len(),
        processed_report_count,
        score_points,
        invalid_report_count,
        false_negative_count,
        precision,
        recall,
        matches,
        invalid_report_ids,
        budget_rejected_report_ids,
        unreported_ground_truth_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth(id: &str, class: &str, x: f64, y: f64, z: f64) -> GroundTruthArtifact {
        GroundTruthArtifact {
            id: id.into(),
            coordinate_frame_id: "darpa".into(),
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
        submission_sequence: u64,
    ) -> CandidateReport {
        CandidateReport {
            id: id.into(),
            coordinate_frame_id: "darpa".into(),
            source_type: class.into(),
            position: Vec3::new(x, y, z),
            submission_sequence,
        }
    }

    fn profile(tolerance: f64, maximum_reports: Option<usize>) -> ReportProfile {
        ReportProfile {
            coordinate_frame_id: "darpa".into(),
            maximum_reports,
            localization_tolerance_m: tolerance,
        }
    }

    fn preliminary_truth() -> Vec<GroundTruthArtifact> {
        (0..20)
            .map(|index| {
                truth(
                    &format!("g{index:02}"),
                    "vent",
                    index as f64 * 20.0,
                    0.0,
                    0.0,
                )
            })
            .collect()
    }

    #[test]
    fn systems_final_requires_a_source_label_mapping() {
        let mut ground_truth = preliminary_truth();
        ground_truth[0].source_type = "unmapped-source-type".into();
        let reports = Vec::new();
        let result = evaluate_systems_final_stream(
            &ground_truth,
            &reports,
            SystemsFinalProfile {
                round: SystemsFinalRound::Preliminary,
                coordinate_frame_id: "darpa".into(),
                type_crosswalk: darpa_final_2021_crosswalk_v0_1(),
            },
        );
        assert!(matches!(
            result,
            Err(RefereeError::MissingSourceTypeMapping(label))
                if label == "unmapped-source-type"
        ));
    }

    #[test]
    fn systems_final_rejects_duplicate_crosswalk_entries() {
        let ground_truth = preliminary_truth();
        let mut crosswalk = darpa_final_2021_crosswalk_v0_1();
        crosswalk.mappings.push(crosswalk.mappings[0].clone());
        let result = evaluate_systems_final_stream(
            &ground_truth,
            &[],
            SystemsFinalProfile {
                round: SystemsFinalRound::Preliminary,
                coordinate_frame_id: "darpa".into(),
                type_crosswalk: crosswalk,
            },
        );
        assert!(matches!(
            result,
            Err(RefereeError::DuplicateSourceTypeMapping(_))
        ));
    }

    #[test]
    fn darpa_crosswalk_preserves_exact_official_report_strings() {
        let crosswalk = darpa_final_2021_crosswalk_v0_1();
        assert_eq!(
            crosswalk.version,
            "darpa-finals-artifact-strings-2021-04-01-v0.1"
        );
        let cellphone = crosswalk
            .mappings
            .iter()
            .find(|entry| entry.source_type == "cellphone")
            .unwrap();
        assert_eq!(cellphone.official_report_string, "Cell Phone");
        let extinguisher = crosswalk
            .mappings
            .iter()
            .find(|entry| entry.source_type == "fireextinguisher")
            .unwrap();
        assert_eq!(extinguisher.official_report_string, "Fire Extinguisher");
    }

    #[test]
    fn systems_preliminary_stream_allows_retry_but_rejects_post_success_duplicate() {
        let ground_truth = preliminary_truth();
        let reports = vec![
            report("r1-wrong-class", "gas", 0.0, 0.0, 0.0, 1),
            report("r2-valid", "Vent", 0.0, 0.0, 0.0, 2),
            report("r3-duplicate", "Vent", 0.0, 0.0, 0.0, 3),
        ];
        let result = evaluate_systems_final_stream(
            &ground_truth,
            &reports,
            SystemsFinalProfile {
                round: SystemsFinalRound::Preliminary,
                coordinate_frame_id: "darpa".into(),
                type_crosswalk: darpa_final_2021_crosswalk_v0_1(),
            },
        )
        .unwrap();
        assert_eq!(result.report_limit, 25);
        assert_eq!(result.score_points, 1);
        assert_eq!(
            result.invalid_report_ids,
            vec!["r1-wrong-class", "r3-duplicate"]
        );
        assert_eq!(result.invalid_report_count, 2);
        assert_eq!(result.false_negative_count, 19);
    }

    #[test]
    fn systems_prize_round_rejects_reports_after_45() {
        let mut ground_truth = preliminary_truth();
        for index in 20..40 {
            ground_truth.push(truth(
                &format!("g{index:02}"),
                "vent",
                index as f64 * 20.0,
                0.0,
                0.0,
            ));
        }
        let reports = (0..46)
            .map(|index| {
                report(
                    &format!("r{index:02}"),
                    "unknown-type",
                    index as f64,
                    500.0,
                    0.0,
                    index,
                )
            })
            .collect::<Vec<_>>();
        let result = evaluate_systems_final_stream(
            &ground_truth,
            &reports,
            SystemsFinalProfile {
                round: SystemsFinalRound::Prize,
                coordinate_frame_id: "darpa".into(),
                type_crosswalk: darpa_final_2021_crosswalk_v0_1(),
            },
        )
        .unwrap();
        assert_eq!(result.report_limit, 45);
        assert_eq!(result.processed_report_count, 45);
        assert_eq!(result.budget_rejected_report_ids.len(), 1);
        assert_eq!(result.budget_rejected_report_ids[0], "r45");
        assert_eq!(result.invalid_report_count, 45);
        assert_eq!(result.score_points, 0);
    }

    #[test]
    fn parses_darpa_tsv_preserving_source_types_and_lineage_ids() {
        let input = "cellphone\t1\t2\t3\t\r\nfireextinguisher\t4.5\t6\t7\r\n";
        let parsed = parse_ground_truth_tsv("prize.csv", "darpa", input).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].id, "prize.csv:1");
        assert_eq!(parsed[0].source_type, "cellphone");
        assert_eq!(parsed[1].source_type, "fireextinguisher");
        assert_eq!(parsed[1].coordinate_frame_id, "darpa");
        assert_eq!(parsed[1].position, Vec3::new(4.5, 6.0, 7.0));
    }

    #[test]
    fn tsv_parser_rejects_malformed_and_non_finite_rows() {
        assert!(matches!(
            parse_ground_truth_tsv("source.csv", "darpa", "vent\t1\t2"),
            Err(GroundTruthParseError::InvalidFieldCount { line: 1, .. })
        ));
        assert!(matches!(
            parse_ground_truth_tsv("source.csv", "darpa", "vent\tNaN\t2\t3"),
            Err(GroundTruthParseError::NonFiniteCoordinate { line: 1, column: 1 })
        ));
        assert!(matches!(
            parse_ground_truth_tsv("source.csv", "darpa", " \t1\t2\t3"),
            Err(GroundTruthParseError::EmptyTypeLabel { line: 1 })
        ));
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
    fn report_budget_uses_submission_order_with_stable_tie_break() {
        let gt = vec![
            truth("g1", "vent", 0.0, 0.0, 0.0),
            truth("g2", "gas", 10.0, 0.0, 0.0),
        ];
        let reports = vec![
            report("r-late", "vent", 0.0, 0.0, 0.0, 3),
            report("r-b", "gas", 10.0, 0.0, 0.0, 1),
            report("r-a", "vent", 0.0, 0.0, 0.0, 2),
        ];
        let result = evaluate_reports(&gt, &reports, profile(1.0, Some(2))).unwrap();
        assert_eq!(result.selected_report_count, 2);
        assert_eq!(result.budget_excluded_report_ids, vec!["r-late"]);
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
        reports.push(report("r1", "vent", 0.1, 0.0, 0.0, 2));
        assert!(matches!(
            evaluate_reports(
                &[truth("g1", "vent", 0.0, 0.0, 0.0)],
                &reports,
                profile(5.0, None)
            ),
            Err(RefereeError::DuplicateCandidateReportId(_))
        ));
    }

    #[test]
    fn duplicate_submission_sequence_is_rejected() {
        let gt = vec![truth("g1", "vent", 0.0, 0.0, 0.0)];
        let reports = vec![
            report("r1", "vent", 0.0, 0.0, 0.0, 7),
            report("r2", "vent", 0.0, 0.0, 0.0, 7),
        ];
        assert_eq!(
            evaluate_reports(&gt, &reports, profile(5.0, None)),
            Err(RefereeError::DuplicateSubmissionSequence(7))
        );
    }

    #[test]
    fn coordinate_frame_mismatch_is_rejected_without_implicit_transform() {
        let gt = vec![truth("g1", "vent", 0.0, 0.0, 0.0)];
        let mut reports = vec![report("r1", "vent", 0.0, 0.0, 0.0, 1)];
        reports[0].coordinate_frame_id = "utm-16n".into();
        assert!(matches!(
            evaluate_reports(&gt, &reports, profile(5.0, None)),
            Err(RefereeError::CoordinateFrameMismatch { .. })
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
