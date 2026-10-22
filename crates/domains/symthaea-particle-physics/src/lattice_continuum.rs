// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Continuum-limit and finite-volume qualification helpers for lattice results.
//!
//! These routines evaluate explicitly declared scaling hypotheses. They do not
//! decide that an `a^2` ansatz is correct, choose which ensembles to drop, or
//! certify that finite-volume effects are negligible.

const FM_TO_GEV_INVERSE: f64 = 5.067_730_716;

#[derive(Debug, Clone, PartialEq)]
pub struct ContinuumDatum {
    pub ensemble_id: String,
    pub observable: String,
    pub scale_setting_id: String,
    pub lattice_spacing_fm: f64,
    pub spatial_extent_sites: usize,
    pub value: f64,
    pub sigma: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContinuumFit {
    /// Continuum value under the declared leading `O(a^2)` ansatz.
    pub intercept: f64,
    pub intercept_sigma: f64,
    /// Coefficient multiplying `a^2`, with `a` measured in fm.
    pub slope_per_fm2: f64,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub chi_square_per_dof: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FiniteVolumeComparison {
    pub smaller_ensemble_id: String,
    pub larger_ensemble_id: String,
    pub smaller_length_fm: f64,
    pub larger_length_fm: f64,
    /// `larger.value - smaller.value`.
    pub absolute_shift: f64,
    pub combined_sigma: f64,
    pub normalized_shift: f64,
    /// Symmetric fractional difference relative to the mean absolute scale.
    pub symmetric_fractional_shift: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContinuumError {
    TooFewPoints,
    InvalidDatum,
    MixedObservable,
    MixedScaleSetting,
    DuplicateLatticeSpacing,
    SingularFit,
    IncompatibleVolumes,
}

fn validate_datum(point: &ContinuumDatum) -> Result<(), ContinuumError> {
    if point.ensemble_id.trim().is_empty()
        || point.observable.trim().is_empty()
        || point.scale_setting_id.trim().is_empty()
        || !point.lattice_spacing_fm.is_finite()
        || point.lattice_spacing_fm <= 0.0
        || point.spatial_extent_sites == 0
        || !point.value.is_finite()
        || !point.sigma.is_finite()
        || point.sigma <= 0.0
    {
        return Err(ContinuumError::InvalidDatum);
    }
    Ok(())
}

/// Weighted least-squares fit to `y(a) = y0 + c a^2`.
///
/// At least three distinct lattice spacings are required. All points must use
/// one observable identifier and one scale-setting lineage so the helper cannot
/// silently combine incompatible quantities.
pub fn fit_leading_a2(points: &[ContinuumDatum]) -> Result<ContinuumFit, ContinuumError> {
    if points.len() < 3 {
        return Err(ContinuumError::TooFewPoints);
    }
    for point in points {
        validate_datum(point)?;
    }

    let observable = &points[0].observable;
    let scale_setting = &points[0].scale_setting_id;
    if points.iter().any(|point| &point.observable != observable) {
        return Err(ContinuumError::MixedObservable);
    }
    if points
        .iter()
        .any(|point| &point.scale_setting_id != scale_setting)
    {
        return Err(ContinuumError::MixedScaleSetting);
    }

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let ai = points[i].lattice_spacing_fm;
            let aj = points[j].lattice_spacing_fm;
            let scale = ai.abs().max(aj.abs()).max(1.0);
            if (ai - aj).abs() <= 1.0e-12 * scale {
                return Err(ContinuumError::DuplicateLatticeSpacing);
            }
        }
    }

    let mut s = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;

    for point in points {
        let x = point.lattice_spacing_fm * point.lattice_spacing_fm;
        let weight = 1.0 / (point.sigma * point.sigma);
        s += weight;
        sx += weight * x;
        sy += weight * point.value;
        sxx += weight * x * x;
        sxy += weight * x * point.value;
    }

    let determinant = s * sxx - sx * sx;
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(ContinuumError::SingularFit);
    }

    let intercept = (sxx * sy - sx * sxy) / determinant;
    let slope = (s * sxy - sx * sy) / determinant;
    let intercept_sigma = (sxx / determinant).sqrt();

    let mut chi_square = 0.0;
    for point in points {
        let x = point.lattice_spacing_fm * point.lattice_spacing_fm;
        let residual = (point.value - (intercept + slope * x)) / point.sigma;
        chi_square += residual * residual;
    }

    let degrees_of_freedom = points.len() - 2;
    Ok(ContinuumFit {
        intercept,
        intercept_sigma,
        slope_per_fm2: slope,
        chi_square,
        degrees_of_freedom,
        chi_square_per_dof: chi_square / degrees_of_freedom as f64,
    })
}

pub fn spatial_length_fm(point: &ContinuumDatum) -> Result<f64, ContinuumError> {
    validate_datum(point)?;
    Ok(point.lattice_spacing_fm * point.spatial_extent_sites as f64)
}

/// Dimensionless `m L` for a mass in GeV and spatial length in fm.
pub fn mass_times_length(mass_gev: f64, length_fm: f64) -> Result<f64, ContinuumError> {
    if !mass_gev.is_finite() || mass_gev <= 0.0 || !length_fm.is_finite() || length_fm <= 0.0 {
        return Err(ContinuumError::InvalidDatum);
    }
    Ok(mass_gev * length_fm * FM_TO_GEV_INVERSE)
}

/// Compare two volumes at the same lattice spacing and scale-setting lineage.
///
/// This reports the observed shift; it deliberately does not decide what
/// magnitude is acceptable for a production physics claim.
pub fn compare_finite_volumes(
    first: &ContinuumDatum,
    second: &ContinuumDatum,
) -> Result<FiniteVolumeComparison, ContinuumError> {
    validate_datum(first)?;
    validate_datum(second)?;
    if first.observable != second.observable || first.scale_setting_id != second.scale_setting_id {
        return Err(ContinuumError::IncompatibleVolumes);
    }
    let spacing_scale = first
        .lattice_spacing_fm
        .abs()
        .max(second.lattice_spacing_fm.abs())
        .max(1.0);
    if (first.lattice_spacing_fm - second.lattice_spacing_fm).abs() > 1.0e-12 * spacing_scale
        || first.spatial_extent_sites == second.spatial_extent_sites
    {
        return Err(ContinuumError::IncompatibleVolumes);
    }

    let (smaller, larger) = if first.spatial_extent_sites < second.spatial_extent_sites {
        (first, second)
    } else {
        (second, first)
    };
    let absolute_shift = larger.value - smaller.value;
    let combined_sigma = (larger.sigma * larger.sigma + smaller.sigma * smaller.sigma).sqrt();
    let mean_scale = 0.5 * (larger.value.abs() + smaller.value.abs());
    let symmetric_fractional_shift = if mean_scale > 0.0 {
        absolute_shift.abs() / mean_scale
    } else {
        0.0
    };

    Ok(FiniteVolumeComparison {
        smaller_ensemble_id: smaller.ensemble_id.clone(),
        larger_ensemble_id: larger.ensemble_id.clone(),
        smaller_length_fm: spatial_length_fm(smaller)?,
        larger_length_fm: spatial_length_fm(larger)?,
        absolute_shift,
        combined_sigma,
        normalized_shift: absolute_shift / combined_sigma,
        symmetric_fractional_shift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(id: &str, a: f64, n: usize, value: f64, sigma: f64) -> ContinuumDatum {
        ContinuumDatum {
            ensemble_id: id.into(),
            observable: "m0pp_over_sqrt_sigma".into(),
            scale_setting_id: "string-tension-v1".into(),
            lattice_spacing_fm: a,
            spatial_extent_sites: n,
            value,
            sigma,
        }
    }

    #[test]
    fn exact_a2_fixture_recovers_continuum_intercept() {
        let points = [
            point("coarse", 0.20, 16, 2.0 + 3.0 * 0.20_f64.powi(2), 0.01),
            point("medium", 0.15, 20, 2.0 + 3.0 * 0.15_f64.powi(2), 0.01),
            point("fine", 0.10, 28, 2.0 + 3.0 * 0.10_f64.powi(2), 0.01),
            point("finer", 0.07, 40, 2.0 + 3.0 * 0.07_f64.powi(2), 0.01),
        ];
        let fit = fit_leading_a2(&points).unwrap();
        assert!((fit.intercept - 2.0).abs() < 1.0e-12);
        assert!((fit.slope_per_fm2 - 3.0).abs() < 1.0e-10);
        assert!(fit.chi_square < 1.0e-20);
        assert_eq!(fit.degrees_of_freedom, 2);
    }

    #[test]
    fn duplicate_lattice_spacing_is_not_fake_continuum_leverage() {
        let points = [
            point("a", 0.1, 24, 1.0, 0.1),
            point("b", 0.1, 32, 1.1, 0.1),
            point("c", 0.2, 16, 1.2, 0.1),
        ];
        assert_eq!(
            fit_leading_a2(&points),
            Err(ContinuumError::DuplicateLatticeSpacing)
        );
    }

    #[test]
    fn volume_comparison_requires_same_spacing_and_lineage() {
        let small = point("small", 0.1, 16, 2.10, 0.02);
        let large = point("large", 0.1, 32, 2.06, 0.02);
        let comparison = compare_finite_volumes(&small, &large).unwrap();
        assert_eq!(comparison.smaller_ensemble_id, "small");
        assert_eq!(comparison.larger_ensemble_id, "large");
        assert!((comparison.absolute_shift + 0.04).abs() < 1.0e-12);
        assert!(comparison.normalized_shift < 0.0);
    }

    #[test]
    fn ml_conversion_is_dimensionless_and_positive() {
        let ml = mass_times_length(1.7, 2.5).unwrap();
        assert!((ml - 1.7 * 2.5 * FM_TO_GEV_INVERSE).abs() < 1.0e-12);
    }
}
