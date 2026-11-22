//! Conservative periodic-velocity Fourier zero-padding and solver refinement.
//!
//! Zero-padding interpolates the currently represented trigonometric polynomial.
//! It does not restore information already lost to aliasing. The caller must
//! check a source-state guard band and independently qualify future response.

use std::fmt;

use super::phase_space_cascade::QuadraticCasimirBudget;
use super::{
    Complex, SolverError, VlasovPoisson1d1v, VlasovPoissonConfig, VlasovPoissonDiagnostics, fft,
};

#[derive(Debug, Clone, PartialEq)]
pub enum SpectralRefinementError {
    InvalidFactor { factor: usize },
    GridSizeOverflow,
    AllocationFailed { elements: usize },
    NonFiniteInput { index: usize },
    NonFiniteInterpolant { index: usize },
    ImaginaryResidualTooLarge { residual: f64, scale: f64 },
    Solver(SolverError),
}

impl fmt::Display for SpectralRefinementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFactor { factor } => {
                write!(
                    formatter,
                    "zero-padding factor {factor} must be a power of two >= 2"
                )
            }
            Self::GridSizeOverflow => {
                write!(formatter, "zero-padded velocity grid size overflowed")
            }
            Self::AllocationFailed { elements } => {
                write!(
                    formatter,
                    "could not allocate {elements} zero-padded values"
                )
            }
            Self::NonFiniteInput { index } => {
                write!(formatter, "zero-padding input {index} is non-finite")
            }
            Self::NonFiniteInterpolant { index } => {
                write!(formatter, "zero-padded interpolant {index} is non-finite")
            }
            Self::ImaginaryResidualTooLarge { residual, scale } => write!(
                formatter,
                "real zero-padded interpolant has imaginary residual {residual:e} at scale {scale:e}"
            ),
            Self::Solver(error) => write!(
                formatter,
                "solver failure during velocity refinement: {error}"
            ),
        }
    }
}

impl std::error::Error for SpectralRefinementError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ZeroPaddedRealLine {
    pub original_len: usize,
    pub padded_len: usize,
    pub factor: usize,
    pub values: Vec<f64>,
    pub maximum_imaginary_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityGridRefinementReport {
    pub old_nv: usize,
    pub new_nv: usize,
    pub refinement_factor: usize,
    pub old_velocity_spacing: f64,
    pub new_velocity_spacing: f64,
    pub simulation_time: f64,
    pub maximum_old_node_interpolation_error: f64,
    pub maximum_interpolant_imaginary_residual: f64,
    pub casimir_before: QuadraticCasimirBudget,
    pub casimir_after: QuadraticCasimirBudget,
    pub diagnostics_before: VlasovPoissonDiagnostics,
    pub diagnostics_after: VlasovPoissonDiagnostics,
    pub total_casimir_relative_delta: f64,
    pub mass_relative_delta: f64,
    pub momentum_delta: f64,
    pub kinetic_energy_relative_delta: f64,
    pub field_energy_relative_delta: f64,
    pub minimum_distribution_before: f64,
    pub minimum_distribution_after: f64,
    pub new_velocity_nyquist_time_mode_1: f64,
    pub new_velocity_recurrence_time_mode_1: f64,
    pub old_state_bytes: usize,
    pub new_state_bytes: usize,
}

fn checked_padded_len(length: usize, factor: usize) -> Result<usize, SpectralRefinementError> {
    if factor < 2 || !factor.is_power_of_two() || length < 4 || !length.is_power_of_two() {
        return Err(SpectralRefinementError::InvalidFactor { factor });
    }
    length
        .checked_mul(factor)
        .ok_or(SpectralRefinementError::GridSizeOverflow)
}

/// Exact trigonometric interpolation of an even, periodic, real-valued line.
///
/// The source Nyquist coefficient is split equally between positive and
/// negative Nyquist frequencies in the larger grid. Every factor-th output
/// sample reproduces the input to floating-point roundoff.
pub fn zero_pad_periodic_real_line(
    samples: &[f64],
    factor: usize,
) -> Result<ZeroPaddedRealLine, SpectralRefinementError> {
    let original_len = samples.len();
    let padded_len = checked_padded_len(original_len, factor)?;
    for (index, value) in samples.iter().enumerate() {
        if !value.is_finite() {
            return Err(SpectralRefinementError::NonFiniteInput { index });
        }
    }

    let mut source_spectrum = Vec::new();
    source_spectrum
        .try_reserve_exact(original_len)
        .map_err(|_| SpectralRefinementError::AllocationFailed {
            elements: original_len,
        })?;
    source_spectrum.extend(
        samples
            .iter()
            .copied()
            .map(|value| Complex::new(value, 0.0)),
    );
    fft(&mut source_spectrum, false);

    let mut padded_spectrum = Vec::new();
    padded_spectrum.try_reserve_exact(padded_len).map_err(|_| {
        SpectralRefinementError::AllocationFailed {
            elements: padded_len,
        }
    })?;
    padded_spectrum.resize(padded_len, Complex::ZERO);

    let scale = factor as f64;
    let source_nyquist = original_len / 2;
    for (mode, coefficient) in source_spectrum.iter().copied().enumerate() {
        if mode == source_nyquist {
            let half = coefficient.scale(0.5 * scale);
            padded_spectrum[source_nyquist] = half;
            padded_spectrum[padded_len - source_nyquist] = half;
        } else {
            let target_mode = if mode < source_nyquist {
                mode
            } else {
                padded_len - (original_len - mode)
            };
            padded_spectrum[target_mode] = coefficient.scale(scale);
        }
    }

    fft(&mut padded_spectrum, true);
    let mut values = Vec::new();
    values.try_reserve_exact(padded_len).map_err(|_| {
        SpectralRefinementError::AllocationFailed {
            elements: padded_len,
        }
    })?;
    let mut maximum_imaginary_residual = 0.0_f64;
    let mut maximum_real_magnitude = 0.0_f64;
    for (index, value) in padded_spectrum.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(SpectralRefinementError::NonFiniteInterpolant { index });
        }
        maximum_imaginary_residual = maximum_imaginary_residual.max(value.im.abs());
        maximum_real_magnitude = maximum_real_magnitude.max(value.re.abs());
        values.push(value.re);
    }
    if maximum_imaginary_residual > 1.0e-11 * maximum_real_magnitude.max(1.0) {
        return Err(SpectralRefinementError::ImaginaryResidualTooLarge {
            residual: maximum_imaginary_residual,
            scale: maximum_real_magnitude,
        });
    }

    Ok(ZeroPaddedRealLine {
        original_len,
        padded_len,
        factor,
        values,
        maximum_imaginary_residual,
    })
}

impl VlasovPoisson1d1v {
    /// Refine the periodic velocity grid by exact Fourier zero-padding.
    ///
    /// The physical interval, x-grid and simulation time are unchanged. A
    /// complete candidate state is built and checked before replacing self.
    /// This cannot recover kinetic content already aliased or discarded.
    pub fn refine_velocity_grid_fourier(
        &mut self,
        factor: usize,
    ) -> Result<VelocityGridRefinementReport, SpectralRefinementError> {
        let old_config = self.config;
        let new_nv = checked_padded_len(old_config.nv, factor)?;
        let new_config = VlasovPoissonConfig {
            nv: new_nv,
            ..old_config
        };
        new_config
            .validate()
            .map_err(SpectralRefinementError::Solver)?;

        let before_casimir = self
            .quadratic_casimir_budget()
            .map_err(SpectralRefinementError::Solver)?;
        let before_diagnostics = self
            .diagnostics()
            .map_err(SpectralRefinementError::Solver)?;

        let new_len = old_config
            .nx
            .checked_mul(new_nv)
            .ok_or(SpectralRefinementError::GridSizeOverflow)?;
        let mut candidate_distribution = Vec::new();
        candidate_distribution
            .try_reserve_exact(new_len)
            .map_err(|_| SpectralRefinementError::AllocationFailed { elements: new_len })?;

        let mut maximum_old_node_interpolation_error = 0.0_f64;
        let mut maximum_interpolant_imaginary_residual = 0.0_f64;
        for row in self.distribution.chunks_exact(old_config.nv) {
            let padded = zero_pad_periodic_real_line(row, factor)?;
            maximum_interpolant_imaginary_residual =
                maximum_interpolant_imaginary_residual.max(padded.maximum_imaginary_residual);
            for (old_index, old_value) in row.iter().copied().enumerate() {
                let reconstructed = padded.values[old_index * factor];
                maximum_old_node_interpolation_error =
                    maximum_old_node_interpolation_error.max((reconstructed - old_value).abs());
            }
            candidate_distribution.extend(padded.values);
        }

        if candidate_distribution.len() != new_len {
            return Err(SpectralRefinementError::Solver(
                SolverError::DistributionLengthMismatch {
                    expected: new_len,
                    actual: candidate_distribution.len(),
                },
            ));
        }

        let mut candidate_solver = VlasovPoisson1d1v::new(new_config, candidate_distribution)
            .map_err(SpectralRefinementError::Solver)?;
        candidate_solver.time = self.time;
        candidate_solver
            .ensure_finite_state()
            .map_err(SpectralRefinementError::Solver)?;

        let after_casimir = candidate_solver
            .quadratic_casimir_budget()
            .map_err(SpectralRefinementError::Solver)?;
        let after_diagnostics = candidate_solver
            .diagnostics()
            .map_err(SpectralRefinementError::Solver)?;

        let total_casimir_relative_delta =
            (after_casimir.total_casimir - before_casimir.total_casimir).abs()
                / before_casimir.total_casimir.abs().max(f64::MIN_POSITIVE);
        let mass_relative_delta = (after_diagnostics.total_mass - before_diagnostics.total_mass)
            .abs()
            / before_diagnostics.total_mass.abs().max(f64::MIN_POSITIVE);
        let momentum_delta = after_diagnostics.momentum - before_diagnostics.momentum;
        let kinetic_energy_relative_delta =
            (after_diagnostics.kinetic_energy - before_diagnostics.kinetic_energy).abs()
                / before_diagnostics
                    .kinetic_energy
                    .abs()
                    .max(f64::MIN_POSITIVE);
        let field_energy_relative_delta =
            (after_diagnostics.field_energy - before_diagnostics.field_energy).abs()
                / before_diagnostics.field_energy.abs().max(f64::MIN_POSITIVE);
        let new_horizon = new_config
            .velocity_sampling_horizon(1)
            .map_err(SpectralRefinementError::Solver)?;

        let old_state_bytes = self
            .distribution
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(SpectralRefinementError::GridSizeOverflow)?;
        let new_state_bytes = candidate_solver
            .distribution
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(SpectralRefinementError::GridSizeOverflow)?;

        let report = VelocityGridRefinementReport {
            old_nv: old_config.nv,
            new_nv,
            refinement_factor: factor,
            old_velocity_spacing: old_config.dv(),
            new_velocity_spacing: new_config.dv(),
            simulation_time: self.time,
            maximum_old_node_interpolation_error,
            maximum_interpolant_imaginary_residual,
            casimir_before: before_casimir,
            casimir_after: after_casimir,
            diagnostics_before: before_diagnostics,
            diagnostics_after: after_diagnostics,
            total_casimir_relative_delta,
            mass_relative_delta,
            momentum_delta,
            kinetic_energy_relative_delta,
            field_energy_relative_delta,
            minimum_distribution_before: before_diagnostics.minimum_distribution,
            minimum_distribution_after: after_diagnostics.minimum_distribution,
            new_velocity_nyquist_time_mode_1: new_horizon.nyquist_encounter_time,
            new_velocity_recurrence_time_mode_1: new_horizon.discrete_recurrence_time,
            old_state_bytes,
            new_state_bytes,
        };

        *self = candidate_solver;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VlasovPoissonConfig, maxwellian_perturbation};

    fn config(nx: usize, nv: usize) -> VlasovPoissonConfig {
        VlasovPoissonConfig {
            nx,
            nv,
            domain_length: 4.0 * std::f64::consts::PI,
            velocity_min: -6.0,
            velocity_max: 6.0,
            charge_to_mass: -1.0,
            permittivity: 1.0,
            background_density: 1.0,
        }
    }

    #[test]
    fn zero_padding_exactly_interpolates_constant_and_signed_modes() {
        let n = 64usize;
        let factor = 4usize;
        let m = n * factor;
        let source: Vec<f64> = (0..n)
            .map(|index| {
                let phase = std::f64::consts::TAU * index as f64 / n as f64;
                1.2 + 0.3 * (3.0 * phase).cos() - 0.2 * (7.0 * phase).sin()
            })
            .collect();
        let padded = zero_pad_periodic_real_line(&source, factor).unwrap();
        assert_eq!(padded.padded_len, m);
        let mut maximum_error = 0.0_f64;
        for index in 0..m {
            let phase = std::f64::consts::TAU * index as f64 / m as f64;
            let expected = 1.2 + 0.3 * (3.0 * phase).cos() - 0.2 * (7.0 * phase).sin();
            maximum_error = maximum_error.max((padded.values[index] - expected).abs());
        }
        assert!(
            maximum_error < 1.0e-12,
            "interpolation error = {maximum_error:e}"
        );
        for (index, source_value) in source.iter().copied().enumerate() {
            assert!((padded.values[index * factor] - source_value).abs() < 1.0e-12);
        }
    }

    #[test]
    fn zero_padding_splits_populated_source_nyquist_symmetrically() {
        let n = 64usize;
        let factor = 2usize;
        let m = n * factor;
        let source: Vec<f64> = (0..n)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let padded = zero_pad_periodic_real_line(&source, factor).unwrap();
        for index in 0..m {
            let expected = (std::f64::consts::PI * index as f64 / factor as f64).cos();
            assert!((padded.values[index] - expected).abs() < 1.0e-12);
        }
        for (index, source_value) in source.iter().copied().enumerate() {
            assert!((padded.values[index * factor] - source_value).abs() < 1.0e-12);
        }
    }

    #[test]
    fn zero_padding_rejects_invalid_factor_and_non_finite_input() {
        let source = vec![0.0; 64];
        assert!(matches!(
            zero_pad_periodic_real_line(&source, 1),
            Err(SpectralRefinementError::InvalidFactor { factor: 1 })
        ));
        let mut non_finite = source;
        non_finite[7] = f64::NAN;
        assert!(matches!(
            zero_pad_periodic_real_line(&non_finite, 2),
            Err(SpectralRefinementError::NonFiniteInput { index: 7 })
        ));
    }

    #[test]
    fn solver_refinement_preserves_old_nodes_moments_c2_field_and_clock() {
        let config = config(64, 128);
        let distribution = maxwellian_perturbation(config, 1.0e-2, 1, 1.0).unwrap();
        let mut solver = VlasovPoisson1d1v::new(config, distribution).unwrap();
        solver.advance_free_streaming(0.125).unwrap();
        let time_before = solver.time();
        let field_before = solver.electric_field().to_vec();
        let report = solver.refine_velocity_grid_fourier(2).unwrap();

        assert_eq!(solver.time(), time_before);
        assert_eq!(solver.config().nv, 256);
        assert_eq!(report.old_nv, 128);
        assert_eq!(report.new_nv, 256);
        assert!(report.maximum_old_node_interpolation_error < 1.0e-12);
        assert!(report.maximum_interpolant_imaginary_residual < 1.0e-12);
        assert!(report.total_casimir_relative_delta < 1.0e-11);
        assert!(report.mass_relative_delta < 1.0e-11);
        assert!(report.kinetic_energy_relative_delta < 1.0e-8);
        assert!(report.field_energy_relative_delta < 1.0e-11);
        assert!((solver.electric_field()[0] - field_before[0]).abs() < 1.0e-12);
        assert_eq!(report.new_state_bytes, 2 * report.old_state_bytes);
    }

    #[test]
    fn refinement_extends_recurrence_horizon_and_prevents_old_grid_return() {
        let config = config(64, 256);
        let distribution = maxwellian_perturbation(config, 1.0e-2, 1, 1.0).unwrap();
        let mut coarse = VlasovPoisson1d1v::new(config, distribution.clone()).unwrap();
        let mut refined = VlasovPoisson1d1v::new(config, distribution.clone()).unwrap();
        let initial = distribution;
        let mut initial_refined = Vec::new();
        for row in initial.chunks_exact(config.nv) {
            initial_refined.extend(zero_pad_periodic_real_line(row, 2).unwrap().values);
        }
        let old_horizon = config.velocity_sampling_horizon(1).unwrap();

        let refine_time = 0.4 * old_horizon.nyquist_encounter_time;
        coarse.advance_free_streaming(refine_time).unwrap();
        refined.advance_free_streaming(refine_time).unwrap();
        let report = refined.refine_velocity_grid_fourier(2).unwrap();

        assert!(
            (report.new_velocity_recurrence_time_mode_1
                - 2.0 * old_horizon.discrete_recurrence_time)
                .abs()
                < 1.0e-10
        );

        let remaining = old_horizon.discrete_recurrence_time - refine_time;
        coarse.advance_free_streaming(remaining).unwrap();
        refined.advance_free_streaming(remaining).unwrap();

        let coarse_error = coarse
            .distribution()
            .iter()
            .zip(initial.iter())
            .map(|(observed, expected)| (observed - expected).abs())
            .fold(0.0_f64, f64::max);
        let refined_old_node_error = refined
            .distribution()
            .iter()
            .step_by(2)
            .zip(initial.iter())
            .map(|(observed, expected)| (observed - expected).abs())
            .fold(0.0_f64, f64::max);
        let refined_full_grid_error = refined
            .distribution()
            .iter()
            .zip(initial_refined.iter())
            .map(|(observed, expected)| (observed - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(coarse_error < 1.0e-10);
        assert!(refined_old_node_error < 1.0e-10);
        assert!(
            refined_full_grid_error > 1.0e-2,
            "refined fine-grid state unexpectedly recurred at the old horizon: {refined_full_grid_error:e}"
        );
    }

    #[test]
    fn invalid_solver_refinement_is_transactional() {
        let config = config(64, 128);
        let distribution = maxwellian_perturbation(config, 1.0e-2, 1, 1.0).unwrap();
        let mut solver = VlasovPoisson1d1v::new(config, distribution).unwrap();
        let before = solver.distribution().to_vec();
        let time = solver.time();
        assert!(matches!(
            solver.refine_velocity_grid_fourier(1),
            Err(SpectralRefinementError::InvalidFactor { factor: 1 })
        ));
        assert_eq!(solver.distribution(), before.as_slice());
        assert_eq!(solver.time(), time);
        assert_eq!(solver.config().nv, config.nv);
    }
}
