// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wilson-loop measurement with smeared spatial and original temporal links.
//!
//! This module implements the LQCD-020C convention:
//!
//! - spatial legs read only from a derived spatial-operator field;
//! - temporal legs read only from the original ensemble field;
//! - backward links use the dagger at the previous site;
//! - only contractible positive-size rectangles are accepted.
//!
//! The operator field therefore cannot silently become an alternate ensemble.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, WilsonGaugeField, su3_dagger, su3_identity, su3_mul, su3_trace,
};

pub const MIXED_SPATIAL_APE_TEMPORAL_UNSMEARED_WILSON_ID: &str =
    "mixed_spatial_ape_temporal_unsmeared_wilson_v1";

#[derive(Debug, Clone, PartialEq)]
pub enum MixedWilsonError {
    Gauge(LatticeGaugeError),
    FieldExtentMismatch {
        original: [usize; 4],
        operator: [usize; 4],
    },
    InvalidSpatialDirection(usize),
    InvalidSpatialExtent(usize),
    InvalidTemporalExtent(usize),
    WindingSpatialExtent {
        requested: usize,
        lattice_extent: usize,
    },
    WindingTemporalExtent {
        requested: usize,
        lattice_extent: usize,
    },
    PathDidNotClose {
        start: Site4,
        end: Site4,
    },
}

impl From<LatticeGaugeError> for MixedWilsonError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

fn validate_fields(
    original: &WilsonGaugeField,
    operator: &WilsonGaugeField,
) -> Result<[usize; 4], MixedWilsonError> {
    let original_dims = original.dims();
    let operator_dims = operator.dims();
    if original_dims != operator_dims {
        return Err(MixedWilsonError::FieldExtentMismatch {
            original: original_dims,
            operator: operator_dims,
        });
    }
    Ok(original_dims)
}

/// Measure one real normalized rectangular Wilson loop.
///
/// The spatial legs use `spatial_operator`; temporal legs always use
/// `original`, even if the operator field contains different temporal links.
pub fn mixed_spatial_wilson_rectangle(
    original: &WilsonGaugeField,
    spatial_operator: &WilsonGaugeField,
    start: Site4,
    spatial_mu: usize,
    spatial_extent: usize,
    temporal_extent: usize,
) -> Result<f64, MixedWilsonError> {
    let dims = validate_fields(original, spatial_operator)?;
    if spatial_mu >= 3 {
        return Err(MixedWilsonError::InvalidSpatialDirection(spatial_mu));
    }
    if spatial_extent == 0 {
        return Err(MixedWilsonError::InvalidSpatialExtent(0));
    }
    if temporal_extent == 0 {
        return Err(MixedWilsonError::InvalidTemporalExtent(0));
    }
    if spatial_extent >= dims[spatial_mu] {
        return Err(MixedWilsonError::WindingSpatialExtent {
            requested: spatial_extent,
            lattice_extent: dims[spatial_mu],
        });
    }
    if temporal_extent >= dims[3] {
        return Err(MixedWilsonError::WindingTemporalExtent {
            requested: temporal_extent,
            lattice_extent: dims[3],
        });
    }

    // Validate the starting site through the canonical field API.
    original.link(start, 0)?;

    let mut product = su3_identity();
    let mut site = start;

    // +spatial: derived operator field.
    for _ in 0..spatial_extent {
        product = su3_mul(&product, spatial_operator.link(site, spatial_mu)?);
        site = original.shift(site, spatial_mu, 1)?;
    }

    // +temporal: original ensemble field only.
    for _ in 0..temporal_extent {
        product = su3_mul(&product, original.link(site, 3)?);
        site = original.shift(site, 3, 1)?;
    }

    // -spatial: move first, then use dagger of the derived spatial link.
    for _ in 0..spatial_extent {
        site = original.shift(site, spatial_mu, -1)?;
        product = su3_mul(
            &product,
            &su3_dagger(spatial_operator.link(site, spatial_mu)?),
        );
    }

    // -temporal: original ensemble field only.
    for _ in 0..temporal_extent {
        site = original.shift(site, 3, -1)?;
        product = su3_mul(&product, &su3_dagger(original.link(site, 3)?));
    }

    if site != start {
        return Err(MixedWilsonError::PathDidNotClose { start, end: site });
    }
    Ok(su3_trace(&product).re / 3.0)
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

/// Average the mixed-link rectangular Wilson loop over every lattice origin.
pub fn average_mixed_spatial_wilson_rectangle(
    original: &WilsonGaugeField,
    spatial_operator: &WilsonGaugeField,
    spatial_mu: usize,
    spatial_extent: usize,
    temporal_extent: usize,
) -> Result<f64, MixedWilsonError> {
    let dims = validate_fields(original, spatial_operator)?;
    let mut sum = 0.0;
    for index in 0..original.site_count() {
        let site = site_from_index(index, dims);
        sum += mixed_spatial_wilson_rectangle(
            original,
            spatial_operator,
            site,
            spatial_mu,
            spatial_extent,
            temporal_extent,
        )?;
    }
    Ok(sum / original.site_count() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_gauge::{Su3Matrix, su3_identity};
    use crate::lattice_spatial_smearing::{SpatialApeConfig, spatial_ape_smear};
    use crate::symmetry_groups::Complex;

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
    fn identity_mixed_loops_are_one() {
        let original = WilsonGaugeField::identity([3, 3, 3, 2]).unwrap();
        let smeared = operator(&original);
        assert!((average_mixed_spatial_wilson_rectangle(&original, &smeared, 0, 1, 1).unwrap() - 1.0).abs() < 1.0e-14);
        assert!((average_mixed_spatial_wilson_rectangle(&original, &smeared, 0, 2, 1).unwrap() - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn matches_independent_lqcd_020c_loop_means() {
        let original = fixture();
        let smeared = operator(&original);
        let expected = [
            (0, 1, 0.999_377_219_425_261_1),
            (0, 2, 0.999_006_848_537_849_8),
            (1, 1, 0.999_415_081_324_206),
            (1, 2, 0.999_233_962_373_143),
            (2, 1, 0.999_452_137_561_139_6),
            (2, 2, 0.999_331_681_542_376_8),
        ];
        for (mu, r, want) in expected {
            let got = average_mixed_spatial_wilson_rectangle(&original, &smeared, mu, r, 1).unwrap();
            assert!((got - want).abs() < 2.0e-12, "mu={mu} r={r}: {got} vs {want}");
        }
    }

    #[test]
    fn operator_temporal_links_are_never_read() {
        let original = fixture();
        let smeared = operator(&original);
        let baseline = mixed_spatial_wilson_rectangle(
            &original,
            &smeared,
            [0, 0, 0, 0],
            0,
            1,
            1,
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
        let got = mixed_spatial_wilson_rectangle(
            &original,
            &tampered,
            [0, 0, 0, 0],
            0,
            1,
            1,
        )
        .unwrap();
        assert_eq!(got, baseline);
    }

    #[test]
    fn mixed_measurement_is_gauge_invariant() {
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
        let transformed_operator = spatial_ape_smear(
            &transformed_original,
            &SpatialApeConfig::default(),
        )
        .unwrap();

        for mu in 0..3 {
            for r in 1..=2 {
                let before = average_mixed_spatial_wilson_rectangle(
                    &original,
                    &smeared,
                    mu,
                    r,
                    1,
                )
                .unwrap();
                let after = average_mixed_spatial_wilson_rectangle(
                    &transformed_original,
                    &transformed_operator,
                    mu,
                    r,
                    1,
                )
                .unwrap();
                assert!((before - after).abs() < 2.0e-12);
            }
        }
    }

    #[test]
    fn winding_and_mismatched_fields_fail_closed() {
        let original = fixture();
        let smeared = operator(&original);
        assert!(matches!(
            average_mixed_spatial_wilson_rectangle(&original, &smeared, 0, 1, 2),
            Err(MixedWilsonError::WindingTemporalExtent { .. })
        ));
        assert!(matches!(
            average_mixed_spatial_wilson_rectangle(&original, &smeared, 0, 3, 1),
            Err(MixedWilsonError::WindingSpatialExtent { .. })
        ));

        let wrong = WilsonGaugeField::identity([3, 3, 2, 2]).unwrap();
        assert!(matches!(
            average_mixed_spatial_wilson_rectangle(&original, &wrong, 0, 1, 1),
            Err(MixedWilsonError::FieldExtentMismatch { .. })
        ));
    }
}
