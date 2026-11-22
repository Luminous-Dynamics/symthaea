//! Held-out audit of whether a covariate sketch preserves conditional null-score laws.
//!
//! This is an empirical two-sample KS + simultaneous DKW diagnostic within predeclared sketch
//! cells. It is deliberately not called the theoretical sketched-calibration leakage Delta_T:
//! it only audits the observed conditional score CDFs for the three pinned RCAF statistics.

use crate::held_out_null_calibration::ESTIMATOR_IDS;
use std::collections::{BTreeMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct SketchScoreObservation {
    pub replicate_id: String,
    pub sketch_cell_id: String,
    /// One non-negative null score per pinned RCAF estimator, in ESTIMATOR_IDS order.
    pub scores: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct SketchSufficiencyConfig {
    pub source_split_id: String,
    pub target_split_id: String,
    /// Sketch cells are declared before inspecting target scores.
    pub expected_sketch_cells: Vec<String>,
    /// Family-wise DKW confidence error across both domains, all cells and all three estimators.
    pub confidence_delta: f64,
    pub minimum_replicates_per_domain_cell: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SketchEstimatorCellAudit {
    pub estimator_id: &'static str,
    pub empirical_ks_distance: Option<f64>,
    pub source_dkw_radius: Option<f64>,
    pub target_dkw_radius: Option<f64>,
    /// min(1, empirical KS + source radius + target radius).
    /// Diagnostic upper envelope for the conditional score-CDF difference.
    pub conditional_score_leakage_upper_bound: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SketchCellAudit {
    pub sketch_cell_id: String,
    pub source_replicates: usize,
    pub target_replicates: usize,
    pub minimum_replicates_per_domain_cell: usize,
    pub disposition: &'static str,
    pub estimators: Vec<SketchEstimatorCellAudit>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OverallSketchEstimatorAudit {
    pub estimator_id: &'static str,
    pub all_expected_cells_audited: bool,
    pub worst_cell_empirical_ks_distance: Option<f64>,
    pub worst_cell_leakage_upper_bound: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnseenTargetSketchCell {
    pub sketch_cell_id: String,
    pub target_replicates: usize,
    pub disposition: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SketchSufficiencyReport {
    pub schema_version: &'static str,
    pub source_status: &'static str,
    pub source_split_id: String,
    pub target_split_id: String,
    pub confidence_delta: f64,
    pub confidence_level: f64,
    pub minimum_replicates_per_domain_cell: usize,
    pub source_replicates: usize,
    pub target_replicates: usize,
    pub expected_sketch_cells: Vec<String>,
    pub unseen_target_sketch_cells: Vec<UnseenTargetSketchCell>,
    pub cells: Vec<SketchCellAudit>,
    pub overall_estimators: Vec<OverallSketchEstimatorAudit>,
}

impl SketchSufficiencyReport {
    pub fn to_json_pretty(&self) -> String {
        let expected = string_array_json(&self.expected_sketch_cells);
        let unseen = self
            .unseen_target_sketch_cells
            .iter()
            .map(|item| {
                format!(
                    "    {{\"sketch_cell_id\": \"{}\", \"target_replicates\": {}, \"disposition\": \"{}\"}}",
                    item.sketch_cell_id, item.target_replicates, item.disposition
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        let cells = self.cells.iter().map(render_cell).collect::<Vec<_>>().join(",\n");
        let overall = self
            .overall_estimators
            .iter()
            .map(|item| {
                format!(
                    concat!(
                        "    {{\n",
                        "      \"estimator_id\": \"{}\",\n",
                        "      \"all_expected_cells_audited\": {},\n",
                        "      \"worst_cell_empirical_ks_distance\": {},\n",
                        "      \"worst_cell_leakage_upper_bound\": {}\n",
                        "    }}"
                    ),
                    item.estimator_id,
                    item.all_expected_cells_audited,
                    option_number_json(item.worst_cell_empirical_ks_distance),
                    option_number_json(item.worst_cell_leakage_upper_bound),
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"source_status\": \"{}\",\n",
                "  \"source_split_id\": \"{}\",\n",
                "  \"target_split_id\": \"{}\",\n",
                "  \"confidence_delta\": {:.12},\n",
                "  \"confidence_level\": {:.12},\n",
                "  \"minimum_replicates_per_domain_cell\": {},\n",
                "  \"source_replicates\": {},\n",
                "  \"target_replicates\": {},\n",
                "  \"expected_sketch_cells\": [{}],\n",
                "  \"unseen_target_sketch_cells\": [\n{}\n  ],\n",
                "  \"cells\": [\n{}\n  ],\n",
                "  \"overall_estimators\": [\n{}\n  ],\n",
                "  \"claim_limits\": [\n",
                "    \"empirical two-sample KS distance plus simultaneous DKW radii; not the theoretical sketched-calibration leakage Delta_T\",\n",
                "    \"validity is conditional on independent or exchangeable observations within each source and target sketch cell\",\n",
                "    \"the audit concerns these three scalar null-score distributions only, not the full conditional response law\",\n",
                "    \"unseen or under-resolved sketch cells do not receive an inferred leakage bound\",\n",
                "    \"synthetic sketch audit only; no empirical scanner/site or biological sufficiency claim\"\n",
                "  ]\n",
                "}}\n"
            ),
            self.schema_version,
            self.source_status,
            self.source_split_id,
            self.target_split_id,
            self.confidence_delta,
            self.confidence_level,
            self.minimum_replicates_per_domain_cell,
            self.source_replicates,
            self.target_replicates,
            expected,
            unseen,
            cells,
            overall,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SketchSufficiencyError {
    InvalidIdentifier(&'static str),
    SourceTargetSplitOverlap,
    EmptyExpectedSketchCells,
    DuplicateExpectedSketchCell,
    InvalidConfidenceDelta,
    InvalidMinimumReplicateCount,
    EmptySourceObservations,
    EmptyTargetObservations,
    DuplicateReplicateId(&'static str),
    SourceTargetReplicateOverlap,
    UnexpectedSourceSketchCell { sketch_cell_id: String },
    InvalidObservation { replicate_id: String },
}

impl fmt::Display for SketchSufficiencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier(field) => {
                write!(f, "{field} must be a non-empty stable ASCII identifier")
            }
            Self::SourceTargetSplitOverlap => {
                write!(f, "source and target split identifiers must be distinct")
            }
            Self::EmptyExpectedSketchCells => {
                write!(f, "at least one predeclared sketch cell is required")
            }
            Self::DuplicateExpectedSketchCell => {
                write!(f, "expected sketch cell identifiers must be unique")
            }
            Self::InvalidConfidenceDelta => {
                write!(f, "confidence delta must be finite and in (0, 0.5)")
            }
            Self::InvalidMinimumReplicateCount => {
                write!(f, "minimum replicates per domain cell must be at least two")
            }
            Self::EmptySourceObservations => {
                write!(f, "source score observations must not be empty")
            }
            Self::EmptyTargetObservations => {
                write!(f, "target score observations must not be empty")
            }
            Self::DuplicateReplicateId(split) => {
                write!(f, "replicate identifiers must be unique within {split}")
            }
            Self::SourceTargetReplicateOverlap => {
                write!(f, "source and target replicate identifiers must be disjoint")
            }
            Self::UnexpectedSourceSketchCell { sketch_cell_id } => {
                write!(f, "source observation belongs to undeclared sketch cell {sketch_cell_id}")
            }
            Self::InvalidObservation { replicate_id } => {
                write!(f, "replicate {replicate_id} has an invalid score vector or sketch cell id")
            }
        }
    }
}

impl std::error::Error for SketchSufficiencyError {}

/// Audit conditional null-score CDFs within predeclared source/target sketch cells.
///
/// For G expected cells, M estimators and family-wise error delta, each domain/cell/estimator CDF receives
/// radius sqrt(log(4GM/delta)/(2n)). The reported upper envelope is the observed two-sample KS
/// distance plus the source and target radii, clipped to one. The union bound covers both domains
/// across both domains, all expected cells and all estimators under within-cell i.i.d./exchangeability. It does not bound arbitrary
/// changes in the full conditional response law or equal the paper's theoretical Delta_T.
pub fn audit_sketch_conditional_score_leakage(
    source: &[SketchScoreObservation],
    target: &[SketchScoreObservation],
    config: &SketchSufficiencyConfig,
) -> Result<SketchSufficiencyReport, SketchSufficiencyError> {
    validate_config(config)?;
    if source.is_empty() {
        return Err(SketchSufficiencyError::EmptySourceObservations);
    }
    if target.is_empty() {
        return Err(SketchSufficiencyError::EmptyTargetObservations);
    }

    let expected = config
        .expected_sketch_cells
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut source_ids = HashSet::with_capacity(source.len());
    let mut source_by_cell = BTreeMap::<String, Vec<&SketchScoreObservation>>::new();
    for item in source {
        validate_identifier(&item.replicate_id, "source_replicate_id")?;
        validate_identifier(&item.sketch_cell_id, "source_sketch_cell_id")?;
        if !source_ids.insert(item.replicate_id.as_str()) {
            return Err(SketchSufficiencyError::DuplicateReplicateId("source"));
        }
        if !expected.contains(item.sketch_cell_id.as_str()) {
            return Err(SketchSufficiencyError::UnexpectedSourceSketchCell {
                sketch_cell_id: item.sketch_cell_id.clone(),
            });
        }
        validate_scores(&item.replicate_id, &item.scores)?;
        source_by_cell
            .entry(item.sketch_cell_id.clone())
            .or_default()
            .push(item);
    }

    let mut target_ids = HashSet::with_capacity(target.len());
    let mut target_by_cell = BTreeMap::<String, Vec<&SketchScoreObservation>>::new();
    for item in target {
        validate_identifier(&item.replicate_id, "target_replicate_id")?;
        validate_identifier(&item.sketch_cell_id, "target_sketch_cell_id")?;
        if !target_ids.insert(item.replicate_id.as_str()) {
            return Err(SketchSufficiencyError::DuplicateReplicateId("target"));
        }
        if source_ids.contains(item.replicate_id.as_str()) {
            return Err(SketchSufficiencyError::SourceTargetReplicateOverlap);
        }
        validate_scores(&item.replicate_id, &item.scores)?;
        target_by_cell
            .entry(item.sketch_cell_id.clone())
            .or_default()
            .push(item);
    }

    let unseen_target_sketch_cells = target_by_cell
        .iter()
        .filter(|(cell, _)| !expected.contains(cell.as_str()))
        .map(|(cell, observations)| UnseenTargetSketchCell {
            sketch_cell_id: cell.clone(),
            target_replicates: observations.len(),
            disposition: "abstain_unseen_sketch_cell",
        })
        .collect::<Vec<_>>();

    let family_radius_factor = (4.0
        * config.expected_sketch_cells.len() as f64
        * ESTIMATOR_IDS.len() as f64
        / config.confidence_delta)
        .ln();
    let mut cells = Vec::with_capacity(config.expected_sketch_cells.len());
    for cell_id in &config.expected_sketch_cells {
        let source_cell = source_by_cell.get(cell_id).map_or(&[][..], |items| items.as_slice());
        let target_cell = target_by_cell.get(cell_id).map_or(&[][..], |items| items.as_slice());
        let source_count = source_cell.len();
        let target_count = target_cell.len();

        let disposition = if source_count == 0 {
            "abstain_missing_source_support"
        } else if target_count == 0 {
            "abstain_missing_target_support"
        } else if source_count < config.minimum_replicates_per_domain_cell
            || target_count < config.minimum_replicates_per_domain_cell
        {
            "abstain_underresolved_sketch_cell"
        } else {
            "audited"
        };

        let mut estimator_audits = Vec::with_capacity(ESTIMATOR_IDS.len());
        for (estimator_index, estimator_id) in ESTIMATOR_IDS.iter().enumerate() {
            if disposition != "audited" {
                estimator_audits.push(SketchEstimatorCellAudit {
                    estimator_id,
                    empirical_ks_distance: None,
                    source_dkw_radius: None,
                    target_dkw_radius: None,
                    conditional_score_leakage_upper_bound: None,
                });
                continue;
            }
            let mut source_scores = source_cell
                .iter()
                .map(|item| item.scores[estimator_index])
                .collect::<Vec<_>>();
            let mut target_scores = target_cell
                .iter()
                .map(|item| item.scores[estimator_index])
                .collect::<Vec<_>>();
            let empirical_ks_distance = empirical_ks_distance(&mut source_scores, &mut target_scores);
            let source_dkw_radius =
                (family_radius_factor / (2.0 * source_count as f64)).sqrt().min(1.0);
            let target_dkw_radius =
                (family_radius_factor / (2.0 * target_count as f64)).sqrt().min(1.0);
            let conditional_score_leakage_upper_bound =
                (empirical_ks_distance + source_dkw_radius + target_dkw_radius).min(1.0);
            estimator_audits.push(SketchEstimatorCellAudit {
                estimator_id,
                empirical_ks_distance: Some(empirical_ks_distance),
                source_dkw_radius: Some(source_dkw_radius),
                target_dkw_radius: Some(target_dkw_radius),
                conditional_score_leakage_upper_bound: Some(
                    conditional_score_leakage_upper_bound,
                ),
            });
        }
        cells.push(SketchCellAudit {
            sketch_cell_id: cell_id.clone(),
            source_replicates: source_count,
            target_replicates: target_count,
            minimum_replicates_per_domain_cell: config.minimum_replicates_per_domain_cell,
            disposition,
            estimators: estimator_audits,
        });
    }

    let overall_estimators = ESTIMATOR_IDS
        .iter()
        .enumerate()
        .map(|(estimator_index, estimator_id)| {
            let all_expected_cells_audited = cells
                .iter()
                .all(|cell| cell.disposition == "audited");
            let worst_cell_empirical_ks_distance = all_expected_cells_audited.then(|| {
                cells
                    .iter()
                    .filter_map(|cell| cell.estimators[estimator_index].empirical_ks_distance)
                    .fold(0.0, f64::max)
            });
            let worst_cell_leakage_upper_bound = all_expected_cells_audited.then(|| {
                cells
                    .iter()
                    .filter_map(|cell| {
                        cell.estimators[estimator_index].conditional_score_leakage_upper_bound
                    })
                    .fold(0.0, f64::max)
            });
            OverallSketchEstimatorAudit {
                estimator_id,
                all_expected_cells_audited,
                worst_cell_empirical_ks_distance,
                worst_cell_leakage_upper_bound,
            }
        })
        .collect();

    Ok(SketchSufficiencyReport {
        schema_version: "rcaf.sketch-sufficiency-audit/0.1.0",
        source_status: "synthetic_sketch_sufficiency_audit",
        source_split_id: config.source_split_id.clone(),
        target_split_id: config.target_split_id.clone(),
        confidence_delta: config.confidence_delta,
        confidence_level: 1.0 - config.confidence_delta,
        minimum_replicates_per_domain_cell: config.minimum_replicates_per_domain_cell,
        source_replicates: source.len(),
        target_replicates: target.len(),
        expected_sketch_cells: config.expected_sketch_cells.clone(),
        unseen_target_sketch_cells,
        cells,
        overall_estimators,
    })
}

fn validate_config(config: &SketchSufficiencyConfig) -> Result<(), SketchSufficiencyError> {
    validate_identifier(&config.source_split_id, "source_split_id")?;
    validate_identifier(&config.target_split_id, "target_split_id")?;
    if config.source_split_id == config.target_split_id {
        return Err(SketchSufficiencyError::SourceTargetSplitOverlap);
    }
    if config.expected_sketch_cells.is_empty() {
        return Err(SketchSufficiencyError::EmptyExpectedSketchCells);
    }
    if !config.confidence_delta.is_finite()
        || !(0.0..0.5).contains(&config.confidence_delta)
    {
        return Err(SketchSufficiencyError::InvalidConfidenceDelta);
    }
    if config.minimum_replicates_per_domain_cell < 2 {
        return Err(SketchSufficiencyError::InvalidMinimumReplicateCount);
    }
    let mut seen = HashSet::with_capacity(config.expected_sketch_cells.len());
    for cell in &config.expected_sketch_cells {
        validate_identifier(cell, "expected_sketch_cell_id")?;
        if !seen.insert(cell.as_str()) {
            return Err(SketchSufficiencyError::DuplicateExpectedSketchCell);
        }
    }
    Ok(())
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), SketchSufficiencyError> {
    if value.trim().is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err(SketchSufficiencyError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_scores(
    replicate_id: &str,
    scores: &[f64; 3],
) -> Result<(), SketchSufficiencyError> {
    if scores.iter().any(|score| !score.is_finite() || *score < 0.0) {
        return Err(SketchSufficiencyError::InvalidObservation {
            replicate_id: replicate_id.to_string(),
        });
    }
    Ok(())
}

/// Supremum of the difference between two empirical CDFs.
pub fn empirical_ks_distance(source: &mut [f64], target: &mut [f64]) -> f64 {
    if source.is_empty()
        || target.is_empty()
        || source.iter().chain(target.iter()).any(|score| !score.is_finite())
    {
        return f64::NAN;
    }
    source.sort_by(f64::total_cmp);
    target.sort_by(f64::total_cmp);
    let mut thresholds = source.to_vec();
    thresholds.extend_from_slice(target);
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|left, right| left.total_cmp(right).is_eq());
    thresholds
        .iter()
        .map(|threshold| {
            let source_count = source.partition_point(|score| score <= threshold);
            let target_count = target.partition_point(|score| score <= threshold);
            (source_count as f64 / source.len() as f64
                - target_count as f64 / target.len() as f64)
                .abs()
        })
        .fold(0.0, f64::max)
}

fn string_array_json(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

fn option_number_json(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.12}"))
        .unwrap_or_else(|| "null".to_string())
}

fn render_cell(cell: &SketchCellAudit) -> String {
    let estimators = cell
        .estimators
        .iter()
        .map(|item| {
            format!(
                concat!(
                    "      {{\n",
                    "        \"estimator_id\": \"{}\",\n",
                    "        \"empirical_ks_distance\": {},\n",
                    "        \"source_dkw_radius\": {},\n",
                    "        \"target_dkw_radius\": {},\n",
                    "        \"conditional_score_leakage_upper_bound\": {}\n",
                    "      }}"
                ),
                item.estimator_id,
                option_number_json(item.empirical_ks_distance),
                option_number_json(item.source_dkw_radius),
                option_number_json(item.target_dkw_radius),
                option_number_json(item.conditional_score_leakage_upper_bound),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        concat!(
            "    {{\n",
            "      \"sketch_cell_id\": \"{}\",\n",
            "      \"source_replicates\": {},\n",
            "      \"target_replicates\": {},\n",
            "      \"minimum_replicates_per_domain_cell\": {},\n",
            "      \"disposition\": \"{}\",\n",
            "      \"estimators\": [\n{}\n      ]\n",
            "    }}"
        ),
        cell.sketch_cell_id,
        cell.source_replicates,
        cell.target_replicates,
        cell.minimum_replicates_per_domain_cell,
        cell.disposition,
        estimators,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SketchSufficiencyConfig {
        SketchSufficiencyConfig {
            source_split_id: "source-null-v1".to_string(),
            target_split_id: "target-null-v1".to_string(),
            expected_sketch_cells: vec![
                "low_motion".to_string(),
                "medium_motion".to_string(),
                "high_motion".to_string(),
            ],
            confidence_delta: 0.05,
            minimum_replicates_per_domain_cell: 30,
        }
    }

    fn source_and_target(medium_shift: f64, high_shift: f64) -> (
        Vec<SketchScoreObservation>,
        Vec<SketchScoreObservation>,
    ) {
        let mut source = Vec::new();
        let mut target = Vec::new();
        for (cell, shift) in [
            ("low_motion", 0.0),
            ("medium_motion", medium_shift),
            ("high_motion", high_shift),
        ] {
            for index in 0..99 {
                let base = (index + 1) as f64 / 100.0;
                source.push(SketchScoreObservation {
                    replicate_id: format!("src-{cell}-{index:03}"),
                    sketch_cell_id: cell.to_string(),
                    scores: [base, base * 0.8, base * 0.6],
                });
                target.push(SketchScoreObservation {
                    replicate_id: format!("tgt-{cell}-{index:03}"),
                    sketch_cell_id: cell.to_string(),
                    scores: [
                        base + shift,
                        base * 0.8 + shift * 0.8,
                        base * 0.6 + shift * 0.6,
                    ],
                });
            }
        }
        (source, target)
    }

    #[test]
    fn identical_conditional_score_laws_have_zero_empirical_ks() {
        let (source, target) = source_and_target(0.0, 0.0);
        let report = audit_sketch_conditional_score_leakage(&source, &target, &config()).unwrap();
        assert_eq!(report.cells.len(), 3);
        assert!(report.cells.iter().all(|cell| cell.disposition == "audited"));
        assert!(report.cells.iter().all(|cell| {
            cell.estimators
                .iter()
                .all(|item| item.empirical_ks_distance == Some(0.0))
        }));
        assert!(report.overall_estimators.iter().all(|item| {
            item.all_expected_cells_audited
                && item.worst_cell_empirical_ks_distance == Some(0.0)
                && item.worst_cell_leakage_upper_bound.is_some()
        }));
    }

    #[test]
    fn conditional_score_shift_is_visible_even_inside_the_same_sketch_cells() {
        let (source, target) = source_and_target(0.10, 0.25);
        let report = audit_sketch_conditional_score_leakage(&source, &target, &config()).unwrap();
        let low = &report.cells[0].estimators[0];
        let medium = &report.cells[1].estimators[0];
        let high = &report.cells[2].estimators[0];
        assert_eq!(low.empirical_ks_distance, Some(0.0));
        assert!(medium.empirical_ks_distance.unwrap() > low.empirical_ks_distance.unwrap());
        assert!(high.empirical_ks_distance.unwrap() > medium.empirical_ks_distance.unwrap());
        assert!(high.conditional_score_leakage_upper_bound.unwrap() >= high.empirical_ks_distance.unwrap());
    }

    #[test]
    fn unseen_target_sketch_cells_are_reported_and_never_borrowed() {
        let (source, mut target) = source_and_target(0.0, 0.0);
        target.extend((0..12).map(|index| SketchScoreObservation {
            replicate_id: format!("tgt-unseen-{index:03}"),
            sketch_cell_id: "new_scanner".to_string(),
            scores: [0.99, 0.79, 0.59],
        }));
        let report = audit_sketch_conditional_score_leakage(&source, &target, &config()).unwrap();
        assert_eq!(report.unseen_target_sketch_cells.len(), 1);
        assert_eq!(report.unseen_target_sketch_cells[0].target_replicates, 12);
        assert_eq!(
            report.unseen_target_sketch_cells[0].disposition,
            "abstain_unseen_sketch_cell"
        );
    }

    #[test]
    fn underresolved_and_missing_domain_cells_have_no_leakage_bound() {
        let (mut source, mut target) = source_and_target(0.0, 0.0);
        source.retain(|item| item.sketch_cell_id != "medium_motion" || item.replicate_id.ends_with("000"));
        target.retain(|item| item.sketch_cell_id != "high_motion");
        let report = audit_sketch_conditional_score_leakage(&source, &target, &config()).unwrap();
        assert_eq!(report.cells[1].disposition, "abstain_underresolved_sketch_cell");
        assert!(report.cells[1].estimators[0].conditional_score_leakage_upper_bound.is_none());
        assert_eq!(report.cells[2].disposition, "abstain_missing_target_support");
        assert!(report.overall_estimators.iter().all(|item| !item.all_expected_cells_audited));
        assert!(report.overall_estimators.iter().all(|item| item.worst_cell_leakage_upper_bound.is_none()));
    }

    #[test]
    fn missing_source_cell_is_not_misreported_as_zero_leakage() {
        let (mut source, target) = source_and_target(0.0, 0.0);
        source.retain(|item| item.sketch_cell_id != "high_motion");
        let report = audit_sketch_conditional_score_leakage(&source, &target, &config()).unwrap();
        assert_eq!(report.cells[2].disposition, "abstain_missing_source_support");
        assert!(report.cells[2].estimators.iter().all(|item| item.empirical_ks_distance.is_none()));
    }

    #[test]
    fn dkw_radius_uses_a_simultaneous_union_bound_over_cells_and_domains() {
        let (source, target) = source_and_target(0.0, 0.0);
        let report = audit_sketch_conditional_score_leakage(&source, &target, &config()).unwrap();
        let expected = ((4.0 * 3.0 * 3.0 / 0.05_f64).ln() / (2.0 * 99.0)).sqrt();
        assert!((report.cells[0].estimators[0].source_dkw_radius.unwrap() - expected).abs() < 1e-12);
        assert!((report.cells[0].estimators[0].target_dkw_radius.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn source_and_target_ids_must_be_disjoint_and_unique() {
        let (source, mut target) = source_and_target(0.0, 0.0);
        target[0].replicate_id = source[0].replicate_id.clone();
        assert_eq!(
            audit_sketch_conditional_score_leakage(&source, &target, &config()),
            Err(SketchSufficiencyError::SourceTargetReplicateOverlap)
        );
        let (source, mut target) = source_and_target(0.0, 0.0);
        target[1].replicate_id = target[0].replicate_id.clone();
        assert_eq!(
            audit_sketch_conditional_score_leakage(&source, &target, &config()),
            Err(SketchSufficiencyError::DuplicateReplicateId("target"))
        );
    }

    #[test]
    fn undeclared_source_cells_and_invalid_scores_fail_closed() {
        let (mut source, target) = source_and_target(0.0, 0.0);
        source[0].sketch_cell_id = "undeclared".to_string();
        assert!(matches!(
            audit_sketch_conditional_score_leakage(&source, &target, &config()),
            Err(SketchSufficiencyError::UnexpectedSourceSketchCell { .. })
        ));
        let (mut source, target) = source_and_target(0.0, 0.0);
        source[0].scores[0] = f64::NAN;
        assert!(matches!(
            audit_sketch_conditional_score_leakage(&source, &target, &config()),
            Err(SketchSufficiencyError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn duplicate_expected_cells_and_invalid_configuration_fail_closed() {
        let (source, target) = source_and_target(0.0, 0.0);
        let mut duplicate = config();
        duplicate.expected_sketch_cells.push("low_motion".to_string());
        assert_eq!(
            audit_sketch_conditional_score_leakage(&source, &target, &duplicate),
            Err(SketchSufficiencyError::DuplicateExpectedSketchCell)
        );
        let mut invalid_delta = config();
        invalid_delta.confidence_delta = 0.0;
        assert_eq!(
            audit_sketch_conditional_score_leakage(&source, &target, &invalid_delta),
            Err(SketchSufficiencyError::InvalidConfidenceDelta)
        );
    }

    #[test]
    fn report_json_exposes_estimators_unseen_cells_and_claim_limits() {
        let (source, mut target) = source_and_target(0.10, 0.25);
        target.push(SketchScoreObservation {
            replicate_id: "tgt-new-001".to_string(),
            sketch_cell_id: "new_scanner".to_string(),
            scores: [0.95, 0.76, 0.57],
        });
        let report = audit_sketch_conditional_score_leakage(&source, &target, &config()).unwrap();
        let json = report.to_json_pretty();
        assert!(json.contains("\"unseen_target_sketch_cells\""));
        assert!(json.contains("absolute_weighted_jaccard"));
        assert!(json.contains("not the theoretical sketched-calibration leakage"));
    }
}
