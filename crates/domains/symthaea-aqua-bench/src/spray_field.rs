//! Deterministic summaries for externally qualified spatial spray-deposition maps.
//!
//! This module does not calibrate instruments, prove capture completeness, assess comfort,
//! assess hygiene or replace standardized showerhead tests. The caller must bind the map
//! to qualified upstream capture-boundary and metrology evidence before setting the
//! completeness flag to true.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// One cell in a versioned bather-plane deposition map.
#[derive(Debug, Clone, PartialEq)]
pub struct DepositionCell {
    pub cell_id: String,
    /// Cell area in square millimetres.
    pub area_mm2: f64,
    /// Water mass collected during the frozen capture interval, in milligrams.
    pub collected_mass_mg: f64,
    /// Whether this cell belongs to the preregistered task target region.
    pub target_region: bool,
}

/// Inputs to the deterministic deposition-map summary.
#[derive(Debug, Clone, Copy)]
pub struct SprayFieldInput<'a> {
    pub cells: &'a [DepositionCell],
    /// Externally selected threshold; Aqua embeds no universal wetting threshold.
    pub target_deposition_threshold_mg_per_cm2: f64,
    /// Must be true only when upstream evidence qualifies the complete spatial capture
    /// boundary and its reconciliation. This boolean is not itself that evidence.
    pub capture_boundary_complete: bool,
}

/// Invalid or incomplete spray-field map input.
#[derive(Debug, Clone, PartialEq)]
pub enum SprayFieldError {
    EmptyGrid,
    UnqualifiedCaptureBoundary,
    InvalidCoverageThreshold,
    MissingCellId,
    DuplicateCellId { cell_id: String },
    InvalidCellArea { cell_id: String },
    InvalidCollectedMass { cell_id: String },
    MissingTargetRegion,
    MissingOffTargetRegion,
    ZeroMappedMass,
    NonFiniteSummary,
}

impl Display for SprayFieldError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGrid => write!(f, "spray field grid must contain at least one cell"),
            Self::UnqualifiedCaptureBoundary => {
                write!(f, "spray field capture boundary is not externally qualified complete")
            }
            Self::InvalidCoverageThreshold => {
                write!(f, "target deposition threshold must be finite and greater than zero")
            }
            Self::MissingCellId => write!(f, "spray field cell identifier is empty"),
            Self::DuplicateCellId { cell_id } => {
                write!(f, "duplicate spray field cell identifier: {cell_id}")
            }
            Self::InvalidCellArea { cell_id } => {
                write!(f, "cell area must be finite and positive: {cell_id}")
            }
            Self::InvalidCollectedMass { cell_id } => {
                write!(f, "collected mass must be finite and non-negative: {cell_id}")
            }
            Self::MissingTargetRegion => write!(f, "spray field has no target-region cells"),
            Self::MissingOffTargetRegion => {
                write!(f, "spray field has no off-target capture cells")
            }
            Self::ZeroMappedMass => write!(f, "spray field mapped mass is zero"),
            Self::NonFiniteSummary => write!(f, "spray field summary overflowed or is non-finite"),
        }
    }
}

impl Error for SprayFieldError {}

/// Descriptive metrics for a qualified, complete mapped deposition field.
///
/// These are separate diagnostics. They are not a comfort score, compliance result or
/// substitute for standard-method coverage and force endpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct SprayFieldSummary {
    pub total_mapped_mass_mg: f64,
    pub target_mapped_mass_mg: f64,
    pub off_target_mapped_mass_mg: f64,
    pub target_area_cm2: f64,
    pub covered_target_area_cm2: f64,
    pub target_area_coverage_fraction: f64,
    pub mean_target_deposition_mg_per_cm2: f64,
    /// None when mean target deposition is zero, so the ratio is undefined.
    pub target_deposition_coefficient_of_variation: Option<f64>,
    /// Fraction of mapped capture mass deposited outside the target region.
    pub off_target_mapped_mass_fraction: f64,
}

/// Summarize a complete spatial deposition map using area-weighted metrics.
///
/// Cell area is converted from mm² to cm². The threshold is supplied by the preregistered
/// test method and is never inferred from the observed candidate or comparator data.
pub fn summarize_deposition_map(
    input: SprayFieldInput<'_>,
) -> Result<SprayFieldSummary, SprayFieldError> {
    if input.cells.is_empty() {
        return Err(SprayFieldError::EmptyGrid);
    }
    if !input.capture_boundary_complete {
        return Err(SprayFieldError::UnqualifiedCaptureBoundary);
    }
    if !input.target_deposition_threshold_mg_per_cm2.is_finite()
        || input.target_deposition_threshold_mg_per_cm2 <= 0.0
    {
        return Err(SprayFieldError::InvalidCoverageThreshold);
    }

    let mut seen = HashSet::with_capacity(input.cells.len());
    let mut total_mapped_mass_mg = 0.0;
    let mut target_mapped_mass_mg = 0.0;
    let mut off_target_mapped_mass_mg = 0.0;
    let mut target_area_cm2 = 0.0;
    let mut covered_target_area_cm2 = 0.0;
    let mut target_cell_count = 0_usize;
    let mut off_target_cell_count = 0_usize;

    for cell in input.cells {
        if cell.cell_id.trim().is_empty() {
            return Err(SprayFieldError::MissingCellId);
        }
        if !seen.insert(cell.cell_id.as_str()) {
            return Err(SprayFieldError::DuplicateCellId {
                cell_id: cell.cell_id.clone(),
            });
        }
        if !cell.area_mm2.is_finite() || cell.area_mm2 <= 0.0 {
            return Err(SprayFieldError::InvalidCellArea {
                cell_id: cell.cell_id.clone(),
            });
        }
        if !cell.collected_mass_mg.is_finite() || cell.collected_mass_mg < 0.0 {
            return Err(SprayFieldError::InvalidCollectedMass {
                cell_id: cell.cell_id.clone(),
            });
        }

        let area_cm2 = cell.area_mm2 / 100.0;
        total_mapped_mass_mg += cell.collected_mass_mg;

        if cell.target_region {
            target_cell_count += 1;
            target_area_cm2 += area_cm2;
            target_mapped_mass_mg += cell.collected_mass_mg;
            let density = cell.collected_mass_mg / area_cm2;
            if density >= input.target_deposition_threshold_mg_per_cm2 {
                covered_target_area_cm2 += area_cm2;
            }
        } else {
            off_target_cell_count += 1;
            off_target_mapped_mass_mg += cell.collected_mass_mg;
        }
    }

    if target_cell_count == 0 {
        return Err(SprayFieldError::MissingTargetRegion);
    }
    if off_target_cell_count == 0 {
        return Err(SprayFieldError::MissingOffTargetRegion);
    }
    if total_mapped_mass_mg <= 0.0 {
        return Err(SprayFieldError::ZeroMappedMass);
    }

    let mean_target_deposition_mg_per_cm2 = target_mapped_mass_mg / target_area_cm2;
    let target_deposition_coefficient_of_variation =
        if mean_target_deposition_mg_per_cm2 > 0.0 {
            let weighted_squared_deviation = input
                .cells
                .iter()
                .filter(|cell| cell.target_region)
                .map(|cell| {
                    let area_cm2 = cell.area_mm2 / 100.0;
                    let density = cell.collected_mass_mg / area_cm2;
                    area_cm2 * (density - mean_target_deposition_mg_per_cm2).powi(2)
                })
                .sum::<f64>();
            let weighted_variance = weighted_squared_deviation / target_area_cm2;
            Some(weighted_variance.sqrt() / mean_target_deposition_mg_per_cm2)
        } else {
            None
        };

    let summary = SprayFieldSummary {
        total_mapped_mass_mg,
        target_mapped_mass_mg,
        off_target_mapped_mass_mg,
        target_area_cm2,
        covered_target_area_cm2,
        target_area_coverage_fraction: covered_target_area_cm2 / target_area_cm2,
        mean_target_deposition_mg_per_cm2,
        target_deposition_coefficient_of_variation,
        off_target_mapped_mass_fraction: off_target_mapped_mass_mg / total_mapped_mass_mg,
    };

    let finite = [
        summary.total_mapped_mass_mg,
        summary.target_mapped_mass_mg,
        summary.off_target_mapped_mass_mg,
        summary.target_area_cm2,
        summary.covered_target_area_cm2,
        summary.target_area_coverage_fraction,
        summary.mean_target_deposition_mg_per_cm2,
        summary.off_target_mapped_mass_fraction,
    ]
    .into_iter()
    .all(f64::is_finite);
    if !finite
        || summary
            .target_deposition_coefficient_of_variation
            .is_some_and(|value| !value.is_finite())
    {
        return Err(SprayFieldError::NonFiniteSummary);
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_cells() -> [DepositionCell; 3] {
        [
            DepositionCell {
                cell_id: "target-left".to_owned(),
                area_mm2: 10_000.0,
                collected_mass_mg: 10.0,
                target_region: true,
            },
            DepositionCell {
                cell_id: "target-right".to_owned(),
                area_mm2: 10_000.0,
                collected_mass_mg: 20.0,
                target_region: true,
            },
            DepositionCell {
                cell_id: "off-target".to_owned(),
                area_mm2: 10_000.0,
                collected_mass_mg: 30.0,
                target_region: false,
            },
        ]
    }

    fn input(cells: &[DepositionCell]) -> SprayFieldInput<'_> {
        SprayFieldInput {
            cells,
            target_deposition_threshold_mg_per_cm2: 0.15,
            capture_boundary_complete: true,
        }
    }

    #[test]
    fn computes_area_weighted_deposition_metrics() {
        let cells = example_cells();
        let summary = summarize_deposition_map(input(&cells)).expect("valid map");

        assert!((summary.total_mapped_mass_mg - 60.0).abs() < 1e-12);
        assert!((summary.target_mapped_mass_mg - 30.0).abs() < 1e-12);
        assert!((summary.off_target_mapped_mass_mg - 30.0).abs() < 1e-12);
        assert!((summary.target_area_cm2 - 200.0).abs() < 1e-12);
        assert!((summary.covered_target_area_cm2 - 100.0).abs() < 1e-12);
        assert!((summary.target_area_coverage_fraction - 0.5).abs() < 1e-12);
        assert!((summary.mean_target_deposition_mg_per_cm2 - 0.15).abs() < 1e-12);
        assert!(
            (summary
                .target_deposition_coefficient_of_variation
                .expect("positive target mean")
                - (1.0 / 3.0))
                .abs()
                < 1e-12
        );
        assert!((summary.off_target_mapped_mass_fraction - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_unqualified_capture_boundary() {
        let cells = example_cells();
        let mut config = input(&cells);
        config.capture_boundary_complete = false;
        assert_eq!(
            summarize_deposition_map(config),
            Err(SprayFieldError::UnqualifiedCaptureBoundary)
        );
    }

    #[test]
    fn rejects_missing_off_target_region() {
        let mut cells = example_cells();
        cells[2].target_region = true;
        assert_eq!(
            summarize_deposition_map(input(&cells)),
            Err(SprayFieldError::MissingOffTargetRegion)
        );
    }

    #[test]
    fn rejects_duplicate_cell_identifiers() {
        let mut cells = example_cells();
        cells[1].cell_id = cells[0].cell_id.clone();
        assert_eq!(
            summarize_deposition_map(input(&cells)),
            Err(SprayFieldError::DuplicateCellId {
                cell_id: "target-left".to_owned()
            })
        );
    }

    #[test]
    fn rejects_non_finite_collected_mass() {
        let mut cells = example_cells();
        cells[0].collected_mass_mg = f64::NAN;
        assert_eq!(
            summarize_deposition_map(input(&cells)),
            Err(SprayFieldError::InvalidCollectedMass {
                cell_id: "target-left".to_owned()
            })
        );
    }

    #[test]
    fn rejects_negative_collected_mass() {
        let mut cells = example_cells();
        cells[0].collected_mass_mg = -0.1;
        assert_eq!(
            summarize_deposition_map(input(&cells)),
            Err(SprayFieldError::InvalidCollectedMass {
                cell_id: "target-left".to_owned()
            })
        );
    }

    #[test]
    fn rejects_non_positive_cell_area() {
        let mut cells = example_cells();
        cells[0].area_mm2 = 0.0;
        assert_eq!(
            summarize_deposition_map(input(&cells)),
            Err(SprayFieldError::InvalidCellArea {
                cell_id: "target-left".to_owned()
            })
        );
    }

    #[test]
    fn rejects_zero_mapped_mass() {
        let mut cells = example_cells();
        for cell in &mut cells {
            cell.collected_mass_mg = 0.0;
        }
        assert_eq!(
            summarize_deposition_map(input(&cells)),
            Err(SprayFieldError::ZeroMappedMass)
        );
    }

    #[test]
    fn rejects_non_positive_threshold() {
        let cells = example_cells();
        let mut config = input(&cells);
        config.target_deposition_threshold_mg_per_cm2 = 0.0;
        assert_eq!(
            summarize_deposition_map(config),
            Err(SprayFieldError::InvalidCoverageThreshold)
        );
    }
}


/// State of one specimen during a controlled spray-field robustness sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprayFieldState {
    Baseline,
    ScaleChallenge,
    PartialBlockage,
    PostClean,
}

/// A deposition-map summary bound to one specimen, geometry and frozen test context.
///
/// The context digest must bind the task region, pressure trace, inlet temperature,
/// ambient conditions, mounting, capture interval, map geometry, measurement protocol
/// and deposition threshold. The geometry digest binds the exact nozzle/CAD revision.
#[derive(Debug, Clone, PartialEq)]
pub struct SprayFieldObservation {
    pub specimen_id: String,
    pub geometry_sha256: String,
    pub test_context_sha256: String,
    pub state: SprayFieldState,
    pub summary: SprayFieldSummary,
}

/// Matched before/after comparison failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SprayFieldComparisonError {
    MissingSpecimenId,
    InvalidGeometryDigest,
    InvalidTestContextDigest,
    SpecimenMismatch,
    GeometryMismatch,
    TestContextMismatch,
    ReferenceMustBeBaseline,
    InvalidSummary,
    NonFiniteDelta,
}

impl Display for SprayFieldComparisonError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSpecimenId => write!(f, "spray-field specimen identifier is empty"),
            Self::InvalidGeometryDigest => write!(f, "spray-field geometry digest is invalid"),
            Self::InvalidTestContextDigest => {
                write!(f, "spray-field test-context digest is invalid")
            }
            Self::SpecimenMismatch => {
                write!(f, "spray-field observations are from different specimens")
            }
            Self::GeometryMismatch => {
                write!(f, "spray-field geometry digest differs between observations")
            }
            Self::TestContextMismatch => {
                write!(f, "spray-field test context differs between observations")
            }
            Self::ReferenceMustBeBaseline => {
                write!(f, "reference spray-field observation must be the baseline state")
            }
            Self::InvalidSummary => write!(f, "spray-field summary is internally invalid"),
            Self::NonFiniteDelta => write!(f, "spray-field comparison delta is non-finite"),
        }
    }
}

impl Error for SprayFieldComparisonError {}

/// Descriptive deltas from the baseline. These values are not pass/fail decisions.
///
/// Coverage and off-target changes are expressed in percentage points. Mean deposition
/// change is a percentage only when baseline target deposition is greater than zero.
/// CV change is a difference in the dimensionless coefficient, not a comfort judgment.
#[derive(Debug, Clone, PartialEq)]
pub struct SprayFieldDelta {
    pub challenged_state: SprayFieldState,
    pub target_coverage_delta_percentage_points: f64,
    pub mean_target_deposition_change_percent: Option<f64>,
    pub target_deposition_cv_delta: Option<f64>,
    pub off_target_mass_fraction_delta_percentage_points: f64,
}

fn valid_spray_field_summary(summary: &SprayFieldSummary) -> bool {
    let finite_nonnegative = [
        summary.total_mapped_mass_mg,
        summary.target_mapped_mass_mg,
        summary.off_target_mapped_mass_mg,
        summary.target_area_cm2,
        summary.covered_target_area_cm2,
        summary.mean_target_deposition_mg_per_cm2,
    ]
    .into_iter()
    .all(|value| value.is_finite() && value >= 0.0);

    let ratios_valid = summary.target_area_coverage_fraction.is_finite()
        && (0.0..=1.0).contains(&summary.target_area_coverage_fraction)
        && summary.off_target_mapped_mass_fraction.is_finite()
        && (0.0..=1.0).contains(&summary.off_target_mapped_mass_fraction);

    let areas_valid = summary.target_area_cm2 > 0.0
        && summary.covered_target_area_cm2 <= summary.target_area_cm2;

    let mass_sum = summary.target_mapped_mass_mg + summary.off_target_mapped_mass_mg;
    let mass_tolerance = 1e-9 * summary.total_mapped_mass_mg.max(1.0);
    let mass_valid = summary.total_mapped_mass_mg > 0.0
        && mass_sum.is_finite()
        && (mass_sum - summary.total_mapped_mass_mg).abs() <= mass_tolerance;

    let coefficient_valid = summary
        .target_deposition_coefficient_of_variation
        .is_none_or(|value| value.is_finite() && value >= 0.0);

    finite_nonnegative
        && ratios_valid
        && areas_valid
        && mass_valid
        && coefficient_valid
}

/// Compare a stress or post-clean state against the same specimen's baseline.
///
/// Both observations must share the exact geometry and a digest of all matched test
/// conditions, including map regions and the preregistered threshold. This function
/// calculates descriptive changes only; it does not decide whether a change is acceptable.
pub fn compare_spray_field_to_baseline(
    baseline: &SprayFieldObservation,
    challenged: &SprayFieldObservation,
) -> Result<SprayFieldDelta, SprayFieldComparisonError> {
    if baseline.specimen_id.trim().is_empty() || challenged.specimen_id.trim().is_empty() {
        return Err(SprayFieldComparisonError::MissingSpecimenId);
    }
    if !is_sha256(&baseline.geometry_sha256) || !is_sha256(&challenged.geometry_sha256) {
        return Err(SprayFieldComparisonError::InvalidGeometryDigest);
    }
    if !is_sha256(&baseline.test_context_sha256)
        || !is_sha256(&challenged.test_context_sha256)
    {
        return Err(SprayFieldComparisonError::InvalidTestContextDigest);
    }
    if baseline.specimen_id != challenged.specimen_id {
        return Err(SprayFieldComparisonError::SpecimenMismatch);
    }
    if baseline.geometry_sha256 != challenged.geometry_sha256 {
        return Err(SprayFieldComparisonError::GeometryMismatch);
    }
    if baseline.test_context_sha256 != challenged.test_context_sha256 {
        return Err(SprayFieldComparisonError::TestContextMismatch);
    }
    if baseline.state != SprayFieldState::Baseline {
        return Err(SprayFieldComparisonError::ReferenceMustBeBaseline);
    }
    if challenged.state == SprayFieldState::Baseline {
        return Err(SprayFieldComparisonError::ReferenceMustBeBaseline);
    }
    if !valid_spray_field_summary(&baseline.summary)
        || !valid_spray_field_summary(&challenged.summary)
    {
        return Err(SprayFieldComparisonError::InvalidSummary);
    }

    let coverage_delta = (challenged.summary.target_area_coverage_fraction
        - baseline.summary.target_area_coverage_fraction)
        * 100.0;
    let off_target_delta = (challenged.summary.off_target_mapped_mass_fraction
        - baseline.summary.off_target_mapped_mass_fraction)
        * 100.0;
    let mean_change = (baseline.summary.mean_target_deposition_mg_per_cm2 > 0.0).then(|| {
        (challenged.summary.mean_target_deposition_mg_per_cm2
            / baseline.summary.mean_target_deposition_mg_per_cm2
            - 1.0)
            * 100.0
    });
    let cv_delta = match (
        baseline.summary.target_deposition_coefficient_of_variation,
        challenged.summary.target_deposition_coefficient_of_variation,
    ) {
        (Some(reference), Some(current)) => Some(current - reference),
        _ => None,
    };

    if !coverage_delta.is_finite()
        || !off_target_delta.is_finite()
        || mean_change.is_some_and(|value| !value.is_finite())
        || cv_delta.is_some_and(|value| !value.is_finite())
    {
        return Err(SprayFieldComparisonError::NonFiniteDelta);
    }

    Ok(SprayFieldDelta {
        challenged_state: challenged.state,
        target_coverage_delta_percentage_points: coverage_delta,
        mean_target_deposition_change_percent: mean_change,
        target_deposition_cv_delta: cv_delta,
        off_target_mass_fraction_delta_percentage_points: off_target_delta,
    })
}

#[cfg(test)]
mod comparison_tests {
    use super::*;

    fn cells(target_a: f64, target_b: f64, off_target: f64) -> [DepositionCell; 3] {
        [
            DepositionCell {
                cell_id: "target-a".to_owned(),
                area_mm2: 10_000.0,
                collected_mass_mg: target_a,
                target_region: true,
            },
            DepositionCell {
                cell_id: "target-b".to_owned(),
                area_mm2: 10_000.0,
                collected_mass_mg: target_b,
                target_region: true,
            },
            DepositionCell {
                cell_id: "off-target".to_owned(),
                area_mm2: 10_000.0,
                collected_mass_mg: off_target,
                target_region: false,
            },
        ]
    }

    fn observation(
        state: SprayFieldState,
        cell_data: &[DepositionCell],
        specimen_id: &str,
        geometry: char,
        context: char,
    ) -> SprayFieldObservation {
        let summary = summarize_deposition_map(SprayFieldInput {
            cells: cell_data,
            target_deposition_threshold_mg_per_cm2: 0.15,
            capture_boundary_complete: true,
        })
        .expect("test deposition map must be valid");

        SprayFieldObservation {
            specimen_id: specimen_id.to_owned(),
            geometry_sha256: std::iter::repeat_n(geometry, 64).collect(),
            test_context_sha256: std::iter::repeat_n(context, 64).collect(),
            state,
            summary,
        }
    }

    #[test]
    fn compares_scale_challenge_to_baseline_without_scoring_acceptability() {
        let base_cells = cells(20.0, 20.0, 10.0);
        let scale_cells = cells(10.0, 10.0, 20.0);
        let baseline = observation(
            SprayFieldState::Baseline,
            &base_cells,
            "specimen-01",
            'a',
            'b',
        );
        let challenged = observation(
            SprayFieldState::ScaleChallenge,
            &scale_cells,
            "specimen-01",
            'a',
            'b',
        );

        let delta =
            compare_spray_field_to_baseline(&baseline, &challenged).expect("matched state pair");

        assert_eq!(delta.challenged_state, SprayFieldState::ScaleChallenge);
        assert!(delta.target_coverage_delta_percentage_points < 0.0);
        assert!(delta
            .off_target_mass_fraction_delta_percentage_points
            > 0.0);
        assert!(delta.mean_target_deposition_change_percent.is_some());
        assert!(delta.target_deposition_cv_delta.is_some());
    }

    #[test]
    fn rejects_different_test_context() {
        let data = cells(20.0, 20.0, 10.0);
        let baseline =
            observation(SprayFieldState::Baseline, &data, "specimen-01", 'a', 'b');
        let challenged =
            observation(SprayFieldState::PostClean, &data, "specimen-01", 'a', 'c');

        assert_eq!(
            compare_spray_field_to_baseline(&baseline, &challenged),
            Err(SprayFieldComparisonError::TestContextMismatch)
        );
    }

    #[test]
    fn rejects_different_specimens() {
        let data = cells(20.0, 20.0, 10.0);
        let baseline =
            observation(SprayFieldState::Baseline, &data, "specimen-01", 'a', 'b');
        let challenged =
            observation(SprayFieldState::PostClean, &data, "specimen-02", 'a', 'b');

        assert_eq!(
            compare_spray_field_to_baseline(&baseline, &challenged),
            Err(SprayFieldComparisonError::SpecimenMismatch)
        );
    }

    #[test]
    fn rejects_different_geometry_revisions() {
        let data = cells(20.0, 20.0, 10.0);
        let baseline =
            observation(SprayFieldState::Baseline, &data, "specimen-01", 'a', 'b');
        let challenged =
            observation(SprayFieldState::PostClean, &data, "specimen-01", 'd', 'b');

        assert_eq!(
            compare_spray_field_to_baseline(&baseline, &challenged),
            Err(SprayFieldComparisonError::GeometryMismatch)
        );
    }

    #[test]
    fn omits_relative_mean_when_baseline_target_deposition_is_zero() {
        let base_cells = cells(0.0, 0.0, 10.0);
        let scale_cells = cells(1.0, 1.0, 8.0);
        let baseline =
            observation(SprayFieldState::Baseline, &base_cells, "specimen-01", 'a', 'b');
        let challenged = observation(
            SprayFieldState::PostClean,
            &scale_cells,
            "specimen-01",
            'a',
            'b',
        );

        let delta =
            compare_spray_field_to_baseline(&baseline, &challenged).expect("matched state pair");
        assert_eq!(delta.mean_target_deposition_change_percent, None);
        assert_eq!(delta.target_deposition_cv_delta, None);
    }

    #[test]
    fn rejects_internally_invalid_public_summary() {
        let data = cells(20.0, 20.0, 10.0);
        let mut baseline =
            observation(SprayFieldState::Baseline, &data, "specimen-01", 'a', 'b');
        baseline.summary.target_area_coverage_fraction = f64::NAN;
        let challenged =
            observation(SprayFieldState::PartialBlockage, &data, "specimen-01", 'a', 'b');

        assert_eq!(
            compare_spray_field_to_baseline(&baseline, &challenged),
            Err(SprayFieldComparisonError::InvalidSummary)
        );
    }
}
