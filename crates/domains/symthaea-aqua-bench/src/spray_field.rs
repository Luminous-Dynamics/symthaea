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
