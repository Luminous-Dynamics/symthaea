// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shortest-path-symmetrized off-axis Wilson-loop measurement.
//!
//! Spatial transport is an arithmetic average of every unique shortest
//! Manhattan path for one declared displacement vector. Spatial edges read
//! only from a derived operator field; temporal edges read only from the
//! original ensemble field. The averaged transporter is gauge covariant but is
//! not asserted to be an SU(3) link.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger, su3_identity, su3_mul,
    su3_trace,
};
use crate::symmetry_groups::Complex;

pub const SHORTEST_PATH_SYMMETRIZED_OFF_AXIS_WILSON_ID: &str =
    "shortest_path_symmetrized_ape_spatial_unsmeared_temporal_wilson_v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffAxisWilsonMeasurement {
    pub value: f64,
    pub path_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OffAxisWilsonError {
    Gauge(LatticeGaugeError),
    FieldExtentMismatch {
        original: [usize; 4],
        operator: [usize; 4],
    },
    ZeroDisplacement,
    WindingSpatialComponent {
        axis: usize,
        requested: usize,
        lattice_extent: usize,
    },
    InvalidTemporalExtent(usize),
    WindingTemporalExtent {
        requested: usize,
        lattice_extent: usize,
    },
    InvalidPathBudget(usize),
    PathBudgetExceeded(usize),
    PathDidNotReachExpectedEndpoint {
        expected: Site4,
        actual: Site4,
    },
    PathCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl From<LatticeGaugeError> for OffAxisWilsonError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

fn validate_fields(
    original: &WilsonGaugeField,
    operator: &WilsonGaugeField,
) -> Result<[usize; 4], OffAxisWilsonError> {
    let original_dims = original.dims();
    let operator_dims = operator.dims();
    if original_dims != operator_dims {
        return Err(OffAxisWilsonError::FieldExtentMismatch {
            original: original_dims,
            operator: operator_dims,
        });
    }
    Ok(original_dims)
}

fn zero_matrix() -> Su3Matrix {
    [[Complex::ZERO; 3]; 3]
}

fn add_assign(target: &mut Su3Matrix, value: &Su3Matrix) {
    for i in 0..3 {
        for j in 0..3 {
            target[i][j] = Complex::new(
                target[i][j].re + value[i][j].re,
                target[i][j].im + value[i][j].im,
            );
        }
    }
}

fn scale_matrix(value: &Su3Matrix, factor: f64) -> Su3Matrix {
    let mut out = zero_matrix();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = Complex::new(value[i][j].re * factor, value[i][j].im * factor);
        }
    }
    out
}

fn expected_endpoint(
    field: &WilsonGaugeField,
    start: Site4,
    displacement: [i32; 3],
) -> Result<Site4, OffAxisWilsonError> {
    let mut site = start;
    for axis in 0..3 {
        site = field.shift(site, axis, displacement[axis] as isize)?;
    }
    Ok(site)
}

fn spatial_edge(
    field: &WilsonGaugeField,
    site: Site4,
    axis: usize,
    direction: i8,
) -> Result<(Su3Matrix, Site4), OffAxisWilsonError> {
    if direction > 0 {
        let edge = field.link(site, axis)?.clone();
        let next = field.shift(site, axis, 1)?;
        Ok((edge, next))
    } else {
        let previous = field.shift(site, axis, -1)?;
        Ok((su3_dagger(field.link(previous, axis)?), previous))
    }
}

struct PathAccumulator<'a> {
    operator: &'a WilsonGaugeField,
    expected_endpoint: Site4,
    directions: [i8; 3],
    max_paths: usize,
    sum: Su3Matrix,
    count: usize,
}

impl PathAccumulator<'_> {
    fn visit(
        &mut self,
        site: Site4,
        remaining: [usize; 3],
        product: &Su3Matrix,
    ) -> Result<(), OffAxisWilsonError> {
        if remaining == [0, 0, 0] {
            if site != self.expected_endpoint {
                return Err(OffAxisWilsonError::PathDidNotReachExpectedEndpoint {
                    expected: self.expected_endpoint,
                    actual: site,
                });
            }
            if self.count >= self.max_paths {
                return Err(OffAxisWilsonError::PathBudgetExceeded(self.max_paths));
            }
            add_assign(&mut self.sum, product);
            self.count += 1;
            return Ok(());
        }

        for axis in 0..3 {
            if remaining[axis] == 0 {
                continue;
            }
            let mut next_remaining = remaining;
            next_remaining[axis] -= 1;
            let (edge, next_site) =
                spatial_edge(self.operator, site, axis, self.directions[axis])?;
            let next_product = su3_mul(product, &edge);
            self.visit(next_site, next_remaining, &next_product)?;
        }
        Ok(())
    }
}

/// Average all unique shortest Manhattan spatial paths for one displacement.
///
/// `max_paths` is an explicit work bound. If the geometry would require more
/// unique path orderings, enumeration stops and fails closed.
pub fn shortest_path_symmetrized_spatial_transporter(
    operator: &WilsonGaugeField,
    start: Site4,
    displacement: [i32; 3],
    max_paths: usize,
) -> Result<(Su3Matrix, Site4, usize), OffAxisWilsonError> {
    if displacement == [0, 0, 0] {
        return Err(OffAxisWilsonError::ZeroDisplacement);
    }
    if max_paths == 0 {
        return Err(OffAxisWilsonError::InvalidPathBudget(max_paths));
    }
    operator.link(start, 0)?;
    let dims = operator.dims();
    let mut remaining = [0usize; 3];
    let mut directions = [0i8; 3];
    for axis in 0..3 {
        let magnitude = displacement[axis].unsigned_abs() as usize;
        if magnitude >= dims[axis] {
            return Err(OffAxisWilsonError::WindingSpatialComponent {
                axis,
                requested: magnitude,
                lattice_extent: dims[axis],
            });
        }
        remaining[axis] = magnitude;
        directions[axis] = if displacement[axis] < 0 { -1 } else { 1 };
    }

    let endpoint = expected_endpoint(operator, start, displacement)?;
    let mut accumulator = PathAccumulator {
        operator,
        expected_endpoint: endpoint,
        directions,
        max_paths,
        sum: zero_matrix(),
        count: 0,
    };
    accumulator.visit(start, remaining, &su3_identity())?;
    let average = scale_matrix(&accumulator.sum, 1.0 / accumulator.count as f64);
    Ok((average, endpoint, accumulator.count))
}

fn temporal_transporter(
    original: &WilsonGaugeField,
    start: Site4,
    temporal_extent: usize,
) -> Result<(Su3Matrix, Site4), OffAxisWilsonError> {
    let dims = original.dims();
    if temporal_extent == 0 {
        return Err(OffAxisWilsonError::InvalidTemporalExtent(temporal_extent));
    }
    if temporal_extent >= dims[3] {
        return Err(OffAxisWilsonError::WindingTemporalExtent {
            requested: temporal_extent,
            lattice_extent: dims[3],
        });
    }
    let mut value = su3_identity();
    let mut site = start;
    for _ in 0..temporal_extent {
        value = su3_mul(&value, original.link(site, 3)?);
        site = original.shift(site, 3, 1)?;
    }
    Ok((value, site))
}

/// Measure one off-axis mixed-link Wilson loop.
pub fn off_axis_mixed_wilson_loop(
    original: &WilsonGaugeField,
    spatial_operator: &WilsonGaugeField,
    start: Site4,
    displacement: [i32; 3],
    temporal_extent: usize,
    max_paths: usize,
) -> Result<OffAxisWilsonMeasurement, OffAxisWilsonError> {
    validate_fields(original, spatial_operator)?;
    original.link(start, 0)?;

    let (bottom, endpoint, path_count) = shortest_path_symmetrized_spatial_transporter(
        spatial_operator,
        start,
        displacement,
        max_paths,
    )?;
    let (temporal_endpoint, top_endpoint) =
        temporal_transporter(original, endpoint, temporal_extent)?;

    let top_start = original.shift(start, 3, temporal_extent as isize)?;
    let (top, top_endpoint_check, top_path_count) = shortest_path_symmetrized_spatial_transporter(
        spatial_operator,
        top_start,
        displacement,
        max_paths,
    )?;
    if top_endpoint_check != top_endpoint {
        return Err(OffAxisWilsonError::PathDidNotReachExpectedEndpoint {
            expected: top_endpoint,
            actual: top_endpoint_check,
        });
    }
    if top_path_count != path_count {
        return Err(OffAxisWilsonError::PathCountMismatch {
            expected: path_count,
            actual: top_path_count,
        });
    }

    let (temporal_start, top_start_check) = temporal_transporter(original, start, temporal_extent)?;
    if top_start_check != top_start {
        return Err(OffAxisWilsonError::PathDidNotReachExpectedEndpoint {
            expected: top_start,
            actual: top_start_check,
        });
    }

    let product = su3_mul(
        &su3_mul(
            &su3_mul(&bottom, &temporal_endpoint),
            &su3_dagger(&top),
        ),
        &su3_dagger(&temporal_start),
    );
    Ok(OffAxisWilsonMeasurement {
        value: su3_trace(&product).re / 3.0,
        path_count,
    })
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

/// Average one off-axis mixed-link Wilson loop over every lattice origin.
pub fn average_off_axis_mixed_wilson_loop(
    original: &WilsonGaugeField,
    spatial_operator: &WilsonGaugeField,
    displacement: [i32; 3],
    temporal_extent: usize,
    max_paths: usize,
) -> Result<OffAxisWilsonMeasurement, OffAxisWilsonError> {
    let dims = validate_fields(original, spatial_operator)?;
    let mut sum = 0.0;
    let mut expected_path_count = None;
    for index in 0..original.site_count() {
        let measured = off_axis_mixed_wilson_loop(
            original,
            spatial_operator,
            site_from_index(index, dims),
            displacement,
            temporal_extent,
            max_paths,
        )?;
        if let Some(expected) = expected_path_count {
            if measured.path_count != expected {
                return Err(OffAxisWilsonError::PathCountMismatch {
                    expected,
                    actual: measured.path_count,
                });
            }
        } else {
            expected_path_count = Some(measured.path_count);
        }
        sum += measured.value;
    }
    Ok(OffAxisWilsonMeasurement {
        value: sum / original.site_count() as f64,
        path_count: expected_path_count.unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_gauge::su3_mul;
    use crate::lattice_mixed_wilson::average_mixed_spatial_wilson_rectangle;
    use crate::lattice_spatial_smearing::{SpatialApeConfig, spatial_ape_smear};

    fn embedded_rotation(pair: (usize, usize), axis: [f64; 3], angle: f64) -> Su3Matrix {
        let norm = axis.iter().map(|value| value * value).sum::<f64>().sqrt();
        let [nx, ny, nz] = axis.map(|value| value / norm);
        let a0 = angle.cos();
        let sine = angle.sin();
        let [a1, a2, a3] = [sine * nx, sine * ny, sine * nz];
        let mut out = su3_identity();
        let (i, j) = pair;
        out[i][i] = Complex::new(a0, a3);
        out[i][j] = Complex::new(a2, a1);
        out[j][i] = Complex::new(-a2, a1);
        out[j][j] = Complex::new(a0, -a3);
        out
    }

    fn fixture() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([3, 3, 3, 2]).unwrap();
        let operations = [
            ([0, 0, 0, 0], 0, (0, 1), [1.0, 2.0, 3.0], 0.31),
            ([1, 0, 1, 0], 1, (0, 2), [2.0, -1.0, 1.0], -0.27),
            ([2, 1, 0, 1], 2, (1, 2), [1.0, 1.0, -2.0], 0.22),
            ([1, 2, 2, 0], 0, (0, 1), [-2.0, 1.0, 1.0], 0.19),
        ];
        for (site, mu, pair, axis, angle) in operations {
            let updated = su3_mul(
                &embedded_rotation(pair, axis, angle),
                field.link(site, mu).unwrap(),
            );
            field.set_link(site, mu, updated).unwrap();
        }
        field
    }

    fn operator(field: &WilsonGaugeField) -> WilsonGaugeField {
        spatial_ape_smear(field, &SpatialApeConfig::default()).unwrap()
    }

    #[test]
    fn reproduces_independent_lqcd_020l_values_and_path_counts() {
        let original = fixture();
        let smeared = operator(&original);
        let expected = [
            ([1, 0, 0], 1, 0.999_377_219_425_261_1),
            ([2, 0, 0], 1, 0.999_006_848_537_849_8),
            ([1, 1, 0], 2, 0.998_868_711_088_126_5),
            ([1, 1, 1], 6, 0.998_619_309_192_763_9),
            ([2, 1, 0], 3, 0.998_413_634_102_571_1),
        ];
        for (displacement, path_count, want) in expected {
            let got = average_off_axis_mixed_wilson_loop(
                &original,
                &smeared,
                displacement,
                1,
                32,
            )
            .unwrap();
            assert_eq!(got.path_count, path_count);
            assert!((got.value - want).abs() < 2.0e-12);
        }
    }

    #[test]
    fn one_path_axis_case_collapses_to_existing_mixed_rectangle() {
        let original = fixture();
        let smeared = operator(&original);
        for r in 1..=2 {
            let off_axis = average_off_axis_mixed_wilson_loop(
                &original,
                &smeared,
                [r as i32, 0, 0],
                1,
                8,
            )
            .unwrap();
            let axis = average_mixed_spatial_wilson_rectangle(&original, &smeared, 0, r, 1)
                .unwrap();
            assert_eq!(off_axis.path_count, 1);
            assert!((off_axis.value - axis).abs() < 1.0e-14);
        }
    }

    #[test]
    fn identity_values_are_one() {
        let original = WilsonGaugeField::identity([3, 3, 3, 2]).unwrap();
        let smeared = operator(&original);
        for displacement in [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0]] {
            let measured = average_off_axis_mixed_wilson_loop(
                &original,
                &smeared,
                displacement,
                1,
                32,
            )
            .unwrap();
            assert!((measured.value - 1.0).abs() < 1.0e-14);
        }
    }

    #[test]
    fn operator_temporal_links_are_never_read() {
        let original = fixture();
        let smeared = operator(&original);
        let baseline = off_axis_mixed_wilson_loop(
            &original,
            &smeared,
            [0, 0, 0, 0],
            [1, 1, 0],
            1,
            8,
        )
        .unwrap();
        let mut tampered = smeared.clone();
        tampered
            .set_link(
                [0, 0, 0, 0],
                3,
                embedded_rotation((0, 1), [1.0, 2.0, 3.0], 0.4),
            )
            .unwrap();
        let got = off_axis_mixed_wilson_loop(
            &original,
            &tampered,
            [0, 0, 0, 0],
            [1, 1, 0],
            1,
            8,
        )
        .unwrap();
        assert_eq!(got, baseline);
    }

    #[test]
    fn measurement_is_gauge_invariant() {
        let original = fixture();
        let smeared = operator(&original);
        let mut gauges = Vec::with_capacity(original.site_count());
        for index in 0..original.site_count() {
            let k = index as f64 + 1.0;
            gauges.push(su3_mul(
                &embedded_rotation((0, 1), [1.0, 2.0, 3.0], 0.017 * k),
                &embedded_rotation((0, 2), [2.0, -1.0, 1.0], -0.011 * k),
            ));
        }
        let transformed_original = original.gauge_transform(&gauges).unwrap();
        let transformed_operator = operator(&transformed_original);
        for displacement in [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0]] {
            let before = average_off_axis_mixed_wilson_loop(
                &original,
                &smeared,
                displacement,
                1,
                32,
            )
            .unwrap();
            let after = average_off_axis_mixed_wilson_loop(
                &transformed_original,
                &transformed_operator,
                displacement,
                1,
                32,
            )
            .unwrap();
            assert_eq!(before.path_count, after.path_count);
            assert!((before.value - after.value).abs() < 2.0e-12);
        }
    }

    #[test]
    fn path_budget_and_winding_fail_closed() {
        let original = fixture();
        let smeared = operator(&original);
        assert_eq!(
            average_off_axis_mixed_wilson_loop(&original, &smeared, [1, 1, 1], 1, 5),
            Err(OffAxisWilsonError::PathBudgetExceeded(5))
        );
        assert!(matches!(
            average_off_axis_mixed_wilson_loop(&original, &smeared, [3, 0, 0], 1, 32),
            Err(OffAxisWilsonError::WindingSpatialComponent { .. })
        ));
        assert_eq!(
            average_off_axis_mixed_wilson_loop(&original, &smeared, [0, 0, 0], 1, 32),
            Err(OffAxisWilsonError::ZeroDisplacement)
        );
    }
}
