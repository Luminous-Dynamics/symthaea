// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded generalized-Bresenham off-axis Wilson transport.
//!
//! This is the production counterpart to independent geometry oracle LQCD-020W.
//! One deterministic 3D Bresenham path is used per cubic orientation, so work is
//! linear in Manhattan separation rather than in the multinomial number of all
//! shortest paths. The exhaustive LQCD-020M operator remains a separate small-
//! vector semantic/measurement convention.
//!
//! Spatial edges read only from the derived operator field. Temporal edges read
//! only from the original ensemble field.

use crate::lattice_cubic_wilson::{CubicOrbitWilsonError, cubic_signed_permutation_orbit};
use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger, su3_identity, su3_mul,
    su3_trace,
};

pub const GENERALIZED_BRESENHAM_WILSON_ID: &str =
    "generalized_bresenham_cubic_ape_spatial_unsmeared_temporal_wilson_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BresenhamStep {
    pub axis: usize,
    pub direction: i8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BresenhamCubicWilsonMeasurement {
    pub value: f64,
    pub orbit_size: usize,
    pub steps_per_orientation: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BresenhamWilsonError {
    Gauge(LatticeGaugeError),
    Cubic(CubicOrbitWilsonError),
    FieldExtentMismatch {
        original: [usize; 4],
        operator: [usize; 4],
    },
    ZeroDisplacement,
    InvalidStepBudget(usize),
    StepBudgetExceeded {
        required: usize,
        budget: usize,
    },
    InvalidOrientationBudget(usize),
    OrbitBudgetExceeded {
        required: usize,
        budget: usize,
    },
    AmbiguousOrWrappingSpatialComponent {
        axis: usize,
        requested: usize,
        lattice_extent: usize,
    },
    InvalidTemporalExtent(usize),
    WindingTemporalExtent {
        requested: usize,
        lattice_extent: usize,
    },
    PathDidNotReachExpectedEndpoint {
        expected: Site4,
        actual: Site4,
    },
    StepCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteAverage,
}

impl From<LatticeGaugeError> for BresenhamWilsonError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

impl From<CubicOrbitWilsonError> for BresenhamWilsonError {
    fn from(value: CubicOrbitWilsonError) -> Self {
        Self::Cubic(value)
    }
}

fn validate_fields(
    original: &WilsonGaugeField,
    operator: &WilsonGaugeField,
) -> Result<[usize; 4], BresenhamWilsonError> {
    let original_dims = original.dims();
    let operator_dims = operator.dims();
    if original_dims != operator_dims {
        return Err(BresenhamWilsonError::FieldExtentMismatch {
            original: original_dims,
            operator: operator_dims,
        });
    }
    Ok(original_dims)
}

fn manhattan_length(displacement: [i32; 3]) -> Result<usize, BresenhamWilsonError> {
    if displacement == [0, 0, 0] {
        return Err(BresenhamWilsonError::ZeroDisplacement);
    }
    displacement.iter().try_fold(0usize, |acc, component| {
        acc.checked_add(component.unsigned_abs() as usize)
            .ok_or(BresenhamWilsonError::StepBudgetExceeded {
                required: usize::MAX,
                budget: usize::MAX,
            })
    })
}

/// Deterministic 3D generalized-Bresenham unit-link sequence.
///
/// The largest-magnitude component is the major axis; magnitude ties are broken
/// by axis index. The two remaining components have independent Bresenham error
/// accumulators. `max_steps` is an explicit allocation/work bound.
pub fn generalized_bresenham_steps(
    displacement: [i32; 3],
    max_steps: usize,
) -> Result<Vec<BresenhamStep>, BresenhamWilsonError> {
    if max_steps == 0 {
        return Err(BresenhamWilsonError::InvalidStepBudget(0));
    }
    let required = manhattan_length(displacement)?;
    if required > max_steps {
        return Err(BresenhamWilsonError::StepBudgetExceeded {
            required,
            budget: max_steps,
        });
    }

    let magnitudes = displacement.map(|component| component.unsigned_abs() as usize);
    let directions = displacement.map(|component| if component < 0 { -1 } else { 1 });
    let mut axes = [0usize, 1, 2];
    axes.sort_by_key(|axis| (std::cmp::Reverse(magnitudes[*axis]), *axis));
    let [major, middle, minor] = axes;

    let major_count = magnitudes[major];
    let middle_count = magnitudes[middle];
    let minor_count = magnitudes[minor];
    let mut chi_middle = 2isize * middle_count as isize - major_count as isize;
    let mut chi_minor = 2isize * minor_count as isize - major_count as isize;
    let mut steps = Vec::with_capacity(required);

    for _ in 0..major_count {
        steps.push(BresenhamStep {
            axis: major,
            direction: directions[major],
        });

        if middle_count > 0 && chi_middle >= 0 {
            steps.push(BresenhamStep {
                axis: middle,
                direction: directions[middle],
            });
            chi_middle -= 2isize * major_count as isize;
        }
        if minor_count > 0 && chi_minor >= 0 {
            steps.push(BresenhamStep {
                axis: minor,
                direction: directions[minor],
            });
            chi_minor -= 2isize * major_count as isize;
        }
        chi_middle += 2isize * middle_count as isize;
        chi_minor += 2isize * minor_count as isize;
    }

    if steps.len() != required {
        return Err(BresenhamWilsonError::StepCountMismatch {
            expected: required,
            actual: steps.len(),
        });
    }
    Ok(steps)
}

fn validate_nonwrapping_spatial_geometry(
    field: &WilsonGaugeField,
    displacement: [i32; 3],
) -> Result<(), BresenhamWilsonError> {
    let dims = field.dims();
    for axis in 0..3 {
        let magnitude = displacement[axis].unsigned_abs() as usize;
        let first_ambiguous = dims[axis] / 2 + dims[axis] % 2;
        if magnitude >= first_ambiguous {
            return Err(BresenhamWilsonError::AmbiguousOrWrappingSpatialComponent {
                axis,
                requested: magnitude,
                lattice_extent: dims[axis],
            });
        }
    }
    Ok(())
}

fn spatial_edge(
    field: &WilsonGaugeField,
    site: Site4,
    step: BresenhamStep,
) -> Result<(Su3Matrix, Site4), BresenhamWilsonError> {
    if step.direction > 0 {
        let edge = *field.link(site, step.axis)?;
        let next = field.shift(site, step.axis, 1)?;
        Ok((edge, next))
    } else {
        let previous = field.shift(site, step.axis, -1)?;
        Ok((su3_dagger(field.link(previous, step.axis)?), previous))
    }
}

fn transport_steps(
    field: &WilsonGaugeField,
    start: Site4,
    steps: &[BresenhamStep],
) -> Result<(Su3Matrix, Site4), BresenhamWilsonError> {
    let mut product = su3_identity();
    let mut site = start;
    for step in steps {
        let (edge, next) = spatial_edge(field, site, *step)?;
        product = su3_mul(&product, &edge);
        site = next;
    }
    Ok((product, site))
}

fn expected_endpoint(
    field: &WilsonGaugeField,
    start: Site4,
    displacement: [i32; 3],
) -> Result<Site4, BresenhamWilsonError> {
    let mut site = start;
    for axis in 0..3 {
        site = field.shift(site, axis, displacement[axis] as isize)?;
    }
    Ok(site)
}

fn temporal_transporter(
    original: &WilsonGaugeField,
    start: Site4,
    temporal_extent: usize,
) -> Result<(Su3Matrix, Site4), BresenhamWilsonError> {
    let dims = original.dims();
    if temporal_extent == 0 {
        return Err(BresenhamWilsonError::InvalidTemporalExtent(0));
    }
    if temporal_extent >= dims[3] {
        return Err(BresenhamWilsonError::WindingTemporalExtent {
            requested: temporal_extent,
            lattice_extent: dims[3],
        });
    }
    let mut product = su3_identity();
    let mut site = start;
    for _ in 0..temporal_extent {
        product = su3_mul(&product, original.link(site, 3)?);
        site = original.shift(site, 3, 1)?;
    }
    Ok((product, site))
}

/// Construct one bounded off-axis spatial transporter from the derived operator field.
pub fn bresenham_spatial_transporter(
    operator: &WilsonGaugeField,
    start: Site4,
    displacement: [i32; 3],
    max_steps: usize,
) -> Result<(Su3Matrix, Site4, usize), BresenhamWilsonError> {
    operator.link(start, 0)?;
    validate_nonwrapping_spatial_geometry(operator, displacement)?;
    let steps = generalized_bresenham_steps(displacement, max_steps)?;
    let (product, endpoint) = transport_steps(operator, start, &steps)?;
    let expected = expected_endpoint(operator, start, displacement)?;
    if endpoint != expected {
        return Err(BresenhamWilsonError::PathDidNotReachExpectedEndpoint {
            expected,
            actual: endpoint,
        });
    }
    Ok((product, endpoint, steps.len()))
}

/// Measure one mixed-link Wilson loop using one bounded Bresenham spatial path.
pub fn bresenham_mixed_wilson_loop(
    original: &WilsonGaugeField,
    spatial_operator: &WilsonGaugeField,
    start: Site4,
    displacement: [i32; 3],
    temporal_extent: usize,
    max_steps: usize,
) -> Result<f64, BresenhamWilsonError> {
    validate_fields(original, spatial_operator)?;
    original.link(start, 0)?;
    validate_nonwrapping_spatial_geometry(spatial_operator, displacement)?;
    let steps = generalized_bresenham_steps(displacement, max_steps)?;

    let (bottom, endpoint) = transport_steps(spatial_operator, start, &steps)?;
    let expected_bottom = expected_endpoint(spatial_operator, start, displacement)?;
    if endpoint != expected_bottom {
        return Err(BresenhamWilsonError::PathDidNotReachExpectedEndpoint {
            expected: expected_bottom,
            actual: endpoint,
        });
    }

    let (right, top_endpoint) = temporal_transporter(original, endpoint, temporal_extent)?;
    let top_start = original.shift(start, 3, temporal_extent as isize)?;
    let (top, top_endpoint_check) = transport_steps(spatial_operator, top_start, &steps)?;
    if top_endpoint_check != top_endpoint {
        return Err(BresenhamWilsonError::PathDidNotReachExpectedEndpoint {
            expected: top_endpoint,
            actual: top_endpoint_check,
        });
    }
    let (left, top_start_check) = temporal_transporter(original, start, temporal_extent)?;
    if top_start_check != top_start {
        return Err(BresenhamWilsonError::PathDidNotReachExpectedEndpoint {
            expected: top_start,
            actual: top_start_check,
        });
    }

    let loop_product = su3_mul(
        &su3_mul(&su3_mul(&bottom, &right), &su3_dagger(&top)),
        &su3_dagger(&left),
    );
    Ok(su3_trace(&loop_product).re / 3.0)
}

fn site_from_index(mut index: usize, dims: [usize; 4]) -> Site4 {
    let t = index % dims[3];
    index /= dims[3];
    let z = index % dims[2];
    index /= dims[2];
    let y = index % dims[1];
    index /= dims[1];
    [index, y, z, t]
}

/// Average one bounded Bresenham mixed-link loop over every lattice origin.
pub fn average_bresenham_mixed_wilson_loop(
    original: &WilsonGaugeField,
    spatial_operator: &WilsonGaugeField,
    displacement: [i32; 3],
    temporal_extent: usize,
    max_steps: usize,
) -> Result<f64, BresenhamWilsonError> {
    let dims = validate_fields(original, spatial_operator)?;
    let mut sum = 0.0;
    for index in 0..original.site_count() {
        sum += bresenham_mixed_wilson_loop(
            original,
            spatial_operator,
            site_from_index(index, dims),
            displacement,
            temporal_extent,
            max_steps,
        )?;
    }
    let value = sum / original.site_count() as f64;
    if !value.is_finite() {
        return Err(BresenhamWilsonError::NonFiniteAverage);
    }
    Ok(value)
}

/// Equal-weight average of the bounded Bresenham observable over a cubic orbit.
pub fn average_cubic_bresenham_mixed_wilson_loop(
    original: &WilsonGaugeField,
    spatial_operator: &WilsonGaugeField,
    representative: [i32; 3],
    temporal_extent: usize,
    max_orientations: usize,
    max_steps_per_orientation: usize,
) -> Result<BresenhamCubicWilsonMeasurement, BresenhamWilsonError> {
    if max_orientations == 0 {
        return Err(BresenhamWilsonError::InvalidOrientationBudget(0));
    }
    let orbit = cubic_signed_permutation_orbit(representative)?;
    if orbit.len() > max_orientations {
        return Err(BresenhamWilsonError::OrbitBudgetExceeded {
            required: orbit.len(),
            budget: max_orientations,
        });
    }

    let expected_steps = manhattan_length(representative)?;
    if expected_steps > max_steps_per_orientation {
        return Err(BresenhamWilsonError::StepBudgetExceeded {
            required: expected_steps,
            budget: max_steps_per_orientation,
        });
    }
    let mut sum = 0.0;
    for orientation in &orbit {
        let steps = manhattan_length(*orientation)?;
        if steps != expected_steps {
            return Err(BresenhamWilsonError::StepCountMismatch {
                expected: expected_steps,
                actual: steps,
            });
        }
        sum += average_bresenham_mixed_wilson_loop(
            original,
            spatial_operator,
            *orientation,
            temporal_extent,
            max_steps_per_orientation,
        )?;
    }

    let value = sum / orbit.len() as f64;
    if !value.is_finite() {
        return Err(BresenhamWilsonError::NonFiniteAverage);
    }
    Ok(BresenhamCubicWilsonMeasurement {
        value,
        orbit_size: orbit.len(),
        steps_per_orientation: expected_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_cubic_wilson::average_cubic_orbit_mixed_wilson_loop;
    use crate::lattice_gauge::su3_diagonal;
    use crate::lattice_spatial_smearing::{SpatialApeConfig, spatial_ape_smear};

    fn axes(steps: &[BresenhamStep]) -> Vec<usize> {
        steps.iter().map(|step| step.axis).collect()
    }

    fn nontrivial_field() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([6, 6, 6, 6]).unwrap();
        field
            .set_link([0, 0, 0, 0], 0, su3_diagonal(0.20, -0.07))
            .unwrap();
        field
            .set_link([2, 1, 3, 1], 1, su3_diagonal(-0.13, 0.09))
            .unwrap();
        field
            .set_link([4, 2, 1, 2], 2, su3_diagonal(0.11, 0.05))
            .unwrap();
        field
            .set_link([1, 4, 2, 3], 3, su3_diagonal(-0.08, 0.03))
            .unwrap();
        field
    }

    fn gauges(dims: [usize; 4]) -> Vec<Su3Matrix> {
        let mut out = Vec::new();
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                for z in 0..dims[2] {
                    for t in 0..dims[3] {
                        out.push(su3_diagonal(
                            0.013 * (x + 2 * y + 3 * z + t) as f64,
                            -0.009 * (1 + x + z + 2 * t) as f64,
                        ));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn reproduces_independent_lqcd_020w_path_sequences() {
        let planar = generalized_bresenham_steps([5, 3, 0], 8).unwrap();
        assert_eq!(axes(&planar), vec![0, 1, 0, 0, 1, 0, 0, 1]);
        let spatial = generalized_bresenham_steps([5, 3, 2], 10).unwrap();
        assert_eq!(axes(&spatial), vec![0, 1, 0, 2, 0, 1, 0, 2, 0, 1]);
        let negative = generalized_bresenham_steps([-5, 3, -2], 10).unwrap();
        assert_eq!(negative[0].direction, -1);
        assert_eq!(negative[1].direction, 1);
        assert_eq!(negative[3].direction, -1);
    }

    #[test]
    fn large_vectors_have_linear_bounded_geometry() {
        let plane = generalized_bresenham_steps([7, 7, 0], 14).unwrap();
        assert_eq!(plane.len(), 14);
        let diagonal = generalized_bresenham_steps([6, 6, 6], 18).unwrap();
        assert_eq!(diagonal.len(), 18);
        assert!(matches!(
            generalized_bresenham_steps([6, 6, 6], 17),
            Err(BresenhamWilsonError::StepBudgetExceeded {
                required: 18,
                budget: 17,
            })
        ));
    }

    #[test]
    fn axis_case_collapses_to_exhaustive_cubic_operator() {
        let original = nontrivial_field();
        let operator = spatial_ape_smear(&original, &SpatialApeConfig::default()).unwrap();
        let bounded = average_cubic_bresenham_mixed_wilson_loop(
            &original,
            &operator,
            [2, 0, 0],
            2,
            6,
            2,
        )
        .unwrap();
        let exhaustive = average_cubic_orbit_mixed_wilson_loop(
            &original,
            &operator,
            [2, 0, 0],
            2,
            6,
            1,
        )
        .unwrap();
        assert_eq!(bounded.orbit_size, 6);
        assert_eq!(bounded.steps_per_orientation, 2);
        assert!((bounded.value - exhaustive.value).abs() < 1.0e-14);
    }

    #[test]
    fn bounded_operator_is_gauge_invariant() {
        let original = nontrivial_field();
        let operator = spatial_ape_smear(&original, &SpatialApeConfig::default()).unwrap();
        let local = gauges(original.dims());
        let transformed_original = original.gauge_transform(&local).unwrap();
        let transformed_operator = operator.gauge_transform(&local).unwrap();

        let before = average_cubic_bresenham_mixed_wilson_loop(
            &original,
            &operator,
            [2, 1, 0],
            2,
            24,
            3,
        )
        .unwrap();
        let after = average_cubic_bresenham_mixed_wilson_loop(
            &transformed_original,
            &transformed_operator,
            [2, 1, 0],
            2,
            24,
            3,
        )
        .unwrap();
        assert!((before.value - after.value).abs() < 1.0e-12);
    }

    #[test]
    fn temporal_links_are_sourced_only_from_original_field() {
        let original = nontrivial_field();
        let operator = spatial_ape_smear(&original, &SpatialApeConfig::default()).unwrap();
        let baseline = average_cubic_bresenham_mixed_wilson_loop(
            &original,
            &operator,
            [2, 1, 0],
            2,
            24,
            3,
        )
        .unwrap();

        let mut tampered = operator.clone();
        tampered
            .set_link([0, 0, 0, 0], 3, su3_diagonal(0.41, -0.17))
            .unwrap();
        let changed = average_cubic_bresenham_mixed_wilson_loop(
            &original,
            &tampered,
            [2, 1, 0],
            2,
            24,
            3,
        )
        .unwrap();
        assert_eq!(baseline.value.to_bits(), changed.value.to_bits());
    }

    #[test]
    fn ambiguous_half_box_and_work_budgets_fail_closed() {
        let original = WilsonGaugeField::identity([16, 16, 16, 32]).unwrap();
        let operator = original.clone();
        assert!(matches!(
            bresenham_spatial_transporter(&operator, [0, 0, 0, 0], [8, 0, 0], 8),
            Err(BresenhamWilsonError::AmbiguousOrWrappingSpatialComponent {
                axis: 0,
                requested: 8,
                lattice_extent: 16,
            })
        ));
        assert!(matches!(
            average_cubic_bresenham_mixed_wilson_loop(
                &original,
                &operator,
                [2, 1, 0],
                1,
                23,
                3,
            ),
            Err(BresenhamWilsonError::OrbitBudgetExceeded {
                required: 24,
                budget: 23,
            })
        ));
    }
}
