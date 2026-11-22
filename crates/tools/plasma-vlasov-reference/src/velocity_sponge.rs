//! Explicit, accounted high-|s| damping in velocity-Fourier space.
//!
//! This is a controlled spectral sponge experiment, not a characteristic
//! outgoing-wave boundary condition. It deliberately removes fine velocity
//! information and reports the resulting quadratic Casimir / f-strophy loss.
//! It must be compared against the collisionless periodic-velocity baseline
//! as a distinct numerical model.

use std::fmt;

use super::phase_space_cascade::QuadraticCasimirBudget;
use super::{Complex, SolverError, VlasovPoisson1d1v, VlasovPoissonDiagnostics, fft, signed_mode};

/// Invalid parameters or solver failures during a velocity-Fourier sponge.
#[derive(Debug, Clone, PartialEq)]
pub enum VelocityFourierSpongeError {
    InvalidDuration,
    InvalidStartFraction,
    InvalidMaximumDamping,
    InvalidExponent,
    NonFiniteTransform,
    InvalidPositivityFloor,
    MomentProjectionSingular,
    MomentResidualExceeded {
        moment: usize,
        relative_residual: f64,
        tolerance: f64,
    },
    PositivityFloorViolation {
        minimum: f64,
        floor: f64,
    },
    Solver(SolverError),
}

impl fmt::Display for VelocityFourierSpongeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDuration => {
                write!(formatter, "sponge duration must be finite and nonnegative")
            }
            Self::InvalidStartFraction => {
                write!(formatter, "sponge start fraction must be in [0, 1)")
            }
            Self::InvalidMaximumDamping => {
                write!(
                    formatter,
                    "maximum damping rate must be finite and nonnegative"
                )
            }
            Self::InvalidExponent => {
                write!(formatter, "sponge exponent must be in 1..=16")
            }
            Self::NonFiniteTransform => {
                write!(
                    formatter,
                    "velocity-Fourier sponge produced a non-finite transform"
                )
            }
            Self::InvalidPositivityFloor => {
                write!(formatter, "positivity floor must be finite and nonnegative")
            }
            Self::MomentProjectionSingular => {
                write!(
                    formatter,
                    "discrete velocity-moment Gram matrix is singular"
                )
            }
            Self::MomentResidualExceeded {
                moment,
                relative_residual,
                tolerance,
            } => write!(
                formatter,
                "moment constraint {moment} residual {relative_residual:e} exceeds tolerance {tolerance:e}"
            ),
            Self::PositivityFloorViolation { minimum, floor } => write!(
                formatter,
                "moment-constrained sponge minimum {minimum:e} is below positivity floor {floor:e}"
            ),
            Self::Solver(error) => write!(formatter, "solver failure during sponge: {error}"),
        }
    }
}

impl std::error::Error for VelocityFourierSpongeError {}

/// Explicit loss accounting for one high-|s| velocity-Fourier sponge step.
///
/// The sponge does not advance the solver clock. Duration is the operator
/// interval used to compute exponential attenuation and is intended to be
/// supplied by an outer split integrator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityFourierSpongeReport {
    pub duration: f64,
    pub start_fraction_of_nyquist: f64,
    pub maximum_damping_rate: f64,
    pub exponent: u32,
    pub nyquist_attenuation: f64,
    pub casimir_before: QuadraticCasimirBudget,
    pub casimir_after: QuadraticCasimirBudget,
    pub diagnostics_before: VlasovPoissonDiagnostics,
    pub diagnostics_after: VlasovPoissonDiagnostics,
    /// Positive is removed C2; small negative roundoff is not clamped.
    pub removed_total_casimir: f64,
    pub removed_spatial_mean_casimir: f64,
    pub removed_fluctuation_f_strophy: f64,
    /// Independent Parseval sum of damped velocity-Fourier modal energy.
    pub spectral_removed_casimir: f64,
    /// (C2 before - C2 after) - spectral modal loss.
    pub casimir_accounting_residual: f64,
    /// Integrated L2 norm of the polynomial moment-restoration correction.
    pub moment_correction_l2: f64,
    pub maximum_pointwise_moment_correction: f64,
    /// Maximum per-row absolute residuals for [mass, momentum, second moment].
    pub max_absolute_moment_residuals: [f64; 3],
    /// Maximum per-row scale-normalized residuals for [mass, momentum, second moment].
    pub max_relative_moment_residuals: [f64; 3],
    pub minimum_distribution_after: f64,
    pub negative_value_fraction_before: f64,
    pub negative_value_fraction_after: f64,
    pub high_s_casimir_before: f64,
    pub high_s_casimir_after: f64,
    /// Positive means high-|s| C2 was removed; negative means reintroduced.
    pub high_s_casimir_removed: f64,
    pub low_s_casimir_before: f64,
    pub low_s_casimir_after: f64,
    /// Signed low-|s| C2 change after moment restoration.
    pub low_s_casimir_change: f64,
    pub preserves_first_three_moments: bool,
}

impl VlasovPoisson1d1v {
    /// Apply an explicit exponential high-|s| sponge to every spatial row.
    ///
    /// For signed velocity-Fourier mode s and normalized mode fraction
    /// r = |s| / s_Nyquist, modes with r <= start_fraction are unchanged.
    /// Above the onset, the amplitude multiplier is
    ///
    ///   exp(-maximum_damping_rate * duration
    ///       * ((r-start)/(1-start))^exponent)
    ///
    /// The transform is periodic on the configured finite velocity interval.
    /// This operation is dissipative, does not advance self.time, and is not
    /// part of step(dt). The caller must explicitly schedule and account for
    /// it. The zero-velocity-Fourier mode is untouched, preserving the
    /// discrete spatial density and therefore the Poisson source up to FFT
    /// roundoff.
    pub fn apply_velocity_fourier_sponge(
        &mut self,
        duration: f64,
        start_fraction_of_nyquist: f64,
        maximum_damping_rate: f64,
        exponent: u32,
    ) -> Result<VelocityFourierSpongeReport, VelocityFourierSpongeError> {
        if !duration.is_finite() || duration < 0.0 {
            return Err(VelocityFourierSpongeError::InvalidDuration);
        }
        if !start_fraction_of_nyquist.is_finite()
            || !(0.0..1.0).contains(&start_fraction_of_nyquist)
        {
            return Err(VelocityFourierSpongeError::InvalidStartFraction);
        }
        if !maximum_damping_rate.is_finite() || maximum_damping_rate < 0.0 {
            return Err(VelocityFourierSpongeError::InvalidMaximumDamping);
        }
        if !(1..=16).contains(&exponent) {
            return Err(VelocityFourierSpongeError::InvalidExponent);
        }

        let before = self
            .quadratic_casimir_budget()
            .map_err(VelocityFourierSpongeError::Solver)?;
        let diagnostics_before = self
            .diagnostics()
            .map_err(VelocityFourierSpongeError::Solver)?;

        // Preserve exact identity semantics and avoid an unnecessary FFT
        // round-trip when the requested operator has no effect.
        if duration == 0.0 || maximum_damping_rate == 0.0 {
            return Ok(VelocityFourierSpongeReport {
                duration,
                start_fraction_of_nyquist,
                maximum_damping_rate,
                exponent,
                nyquist_attenuation: 1.0,
                casimir_before: before,
                casimir_after: before,
                diagnostics_before,
                diagnostics_after: diagnostics_before,
                removed_total_casimir: 0.0,
                removed_spatial_mean_casimir: 0.0,
                removed_fluctuation_f_strophy: 0.0,
                spectral_removed_casimir: 0.0,
                casimir_accounting_residual: 0.0,
                moment_correction_l2: 0.0,
                maximum_pointwise_moment_correction: 0.0,
                max_absolute_moment_residuals: [0.0; 3],
                max_relative_moment_residuals: [0.0; 3],
                minimum_distribution_after: diagnostics_before.minimum_distribution,
                negative_value_fraction_before: negative_value_fraction(&self.distribution),
                negative_value_fraction_after: negative_value_fraction(&self.distribution),
                high_s_casimir_before: 0.0,
                high_s_casimir_after: 0.0,
                high_s_casimir_removed: 0.0,
                low_s_casimir_before: 0.0,
                low_s_casimir_after: 0.0,
                low_s_casimir_change: 0.0,
                preserves_first_three_moments: true,
            });
        }

        let config = self.config;
        let expected =
            config
                .nx
                .checked_mul(config.nv)
                .ok_or(VelocityFourierSpongeError::Solver(
                    SolverError::GridSizeOverflow,
                ))?;

        let mut candidate = Vec::new();
        candidate.try_reserve_exact(expected).map_err(|_| {
            VelocityFourierSpongeError::Solver(SolverError::AllocationFailed { elements: expected })
        })?;

        let mut line = Vec::new();
        line.try_reserve_exact(config.nv).map_err(|_| {
            VelocityFourierSpongeError::Solver(SolverError::AllocationFailed {
                elements: config.nv,
            })
        })?;
        line.resize(config.nv, Complex::ZERO);

        let nyquist_mode = config.nv / 2;
        let mut spectral_removed_casimir = 0.0;
        let mut high_s_casimir_before = 0.0;
        let mut high_s_casimir_after = 0.0;
        let mut low_s_casimir_before = 0.0;
        let mut low_s_casimir_after = 0.0;
        let spectral_weight = config.dx() * config.dv() / config.nv as f64;

        for ix in 0..config.nx {
            let row_start = ix * config.nv;
            for (mode, value) in line.iter_mut().enumerate() {
                *value = Complex::new(self.distribution[row_start + mode], 0.0);
            }

            fft(&mut line, false);
            for (mode, coefficient) in line.iter_mut().enumerate() {
                let signed = signed_mode(mode, config.nv);
                let normalized = signed.unsigned_abs() as f64 / nyquist_mode as f64;
                let attenuation = if normalized > start_fraction_of_nyquist {
                    let ramp = (normalized - start_fraction_of_nyquist)
                        / (1.0 - start_fraction_of_nyquist);
                    let damping_rate = maximum_damping_rate * ramp.powi(exponent as i32);
                    (-damping_rate * duration).exp()
                } else {
                    1.0
                };

                let power = coefficient
                    .re
                    .mul_add(coefficient.re, coefficient.im * coefficient.im);
                let attenuated_power = attenuation * attenuation * power;
                spectral_removed_casimir +=
                    (1.0 - attenuation * attenuation) * power * spectral_weight;
                if normalized > start_fraction_of_nyquist {
                    high_s_casimir_before += power * spectral_weight;
                    high_s_casimir_after += attenuated_power * spectral_weight;
                } else {
                    low_s_casimir_before += power * spectral_weight;
                    low_s_casimir_after += attenuated_power * spectral_weight;
                }
                *coefficient = coefficient.scale(attenuation);
                if !coefficient.re.is_finite()
                    || !coefficient.im.is_finite()
                    || !spectral_removed_casimir.is_finite()
                {
                    return Err(VelocityFourierSpongeError::NonFiniteTransform);
                }
            }

            fft(&mut line, true);
            for value in &line {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(VelocityFourierSpongeError::NonFiniteTransform);
                }
                candidate.push(value.re);
            }
        }

        let previous_distribution = self.distribution.clone();
        let previous_field = self.electric_field.clone();
        self.distribution = candidate;

        if let Err(error) = self
            .refresh_electric_field()
            .and_then(|()| self.ensure_finite_state())
        {
            self.distribution = previous_distribution;
            self.electric_field = previous_field;
            return Err(VelocityFourierSpongeError::Solver(error));
        }

        let after = match self.quadratic_casimir_budget() {
            Ok(budget) => budget,
            Err(error) => {
                self.distribution = previous_distribution;
                self.electric_field = previous_field;
                return Err(VelocityFourierSpongeError::Solver(error));
            }
        };
        let diagnostics_after = match self.diagnostics() {
            Ok(diagnostics) => diagnostics,
            Err(error) => {
                self.distribution = previous_distribution;
                self.electric_field = previous_field;
                return Err(VelocityFourierSpongeError::Solver(error));
            }
        };

        let removed_total_casimir = before.total_casimir - after.total_casimir;
        let removed_spatial_mean_casimir = before.spatial_mean_casimir - after.spatial_mean_casimir;
        let removed_fluctuation_f_strophy =
            before.fluctuation_f_strophy - after.fluctuation_f_strophy;
        let casimir_accounting_residual = removed_total_casimir - spectral_removed_casimir;

        if !removed_total_casimir.is_finite()
            || !removed_spatial_mean_casimir.is_finite()
            || !removed_fluctuation_f_strophy.is_finite()
            || !spectral_removed_casimir.is_finite()
            || !casimir_accounting_residual.is_finite()
        {
            self.distribution = previous_distribution;
            self.electric_field = previous_field;
            return Err(VelocityFourierSpongeError::NonFiniteTransform);
        }

        let nyquist_attenuation = (-maximum_damping_rate * duration).exp();

        Ok(VelocityFourierSpongeReport {
            duration,
            start_fraction_of_nyquist,
            maximum_damping_rate,
            exponent,
            nyquist_attenuation,
            casimir_before: before,
            casimir_after: after,
            diagnostics_before,
            diagnostics_after,
            removed_total_casimir,
            removed_spatial_mean_casimir,
            removed_fluctuation_f_strophy,
            spectral_removed_casimir,
            casimir_accounting_residual,
            moment_correction_l2: 0.0,
            maximum_pointwise_moment_correction: 0.0,
            max_absolute_moment_residuals: [0.0; 3],
            max_relative_moment_residuals: [0.0; 3],
            minimum_distribution_after: diagnostics_after.minimum_distribution,
            negative_value_fraction_before: negative_value_fraction(&previous_distribution),
            negative_value_fraction_after: negative_value_fraction(&self.distribution),
            high_s_casimir_before,
            high_s_casimir_after,
            high_s_casimir_removed: high_s_casimir_before - high_s_casimir_after,
            low_s_casimir_before,
            low_s_casimir_after,
            low_s_casimir_change: low_s_casimir_after - low_s_casimir_before,
            preserves_first_three_moments: false,
        })
    }
}

/// Discrete moments [integral f, integral v f, integral v^2 f] on one velocity row.
const MOMENT_PROJECTION_RELATIVE_TOLERANCE: f64 = 1.0e-11;

fn negative_value_fraction(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().filter(|value| **value < 0.0).count() as f64 / values.len() as f64
}

fn discrete_velocity_moments(row: &[f64], config: super::VlasovPoissonConfig) -> [f64; 3] {
    let mut moments = [0.0; 3];
    for (iv, value) in row.iter().copied().enumerate() {
        let velocity = config.velocity(iv);
        let weighted = value * config.dv();
        moments[0] += weighted;
        moments[1] = velocity.mul_add(weighted, moments[1]);
        moments[2] = (velocity * velocity).mul_add(weighted, moments[2]);
    }
    moments
}

fn solve_three_by_three(
    gram: [[f64; 3]; 3],
    rhs: [f64; 3],
) -> Result<[f64; 3], VelocityFourierSpongeError> {
    let mut augmented = [[0.0; 4]; 3];
    for row in 0..3 {
        for column in 0..3 {
            augmented[row][column] = gram[row][column];
        }
        augmented[row][3] = rhs[row];
    }

    for column in 0..3 {
        let pivot_row = (column..3)
            .max_by(|left, right| {
                augmented[*left][column]
                    .abs()
                    .total_cmp(&augmented[*right][column].abs())
            })
            .ok_or(VelocityFourierSpongeError::MomentProjectionSingular)?;
        let pivot = augmented[pivot_row][column];
        let row_scale = augmented[pivot_row][..3]
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if !pivot.is_finite()
            || pivot.abs() <= 64.0 * f64::EPSILON * row_scale.max(f64::MIN_POSITIVE)
        {
            return Err(VelocityFourierSpongeError::MomentProjectionSingular);
        }
        if pivot_row != column {
            augmented.swap(pivot_row, column);
        }

        let diagonal = augmented[column][column];
        for value in &mut augmented[column][column..4] {
            *value /= diagonal;
        }
        let pivot_values = augmented[column];
        for (row_index, row_values) in augmented.iter_mut().enumerate() {
            if row_index == column {
                continue;
            }
            let factor = row_values[column];
            for (entry, pivot_entry) in row_values[column..]
                .iter_mut()
                .zip(pivot_values[column..].iter())
            {
                *entry -= factor * pivot_entry;
            }
        }
    }

    let solution = [augmented[0][3], augmented[1][3], augmented[2][3]];
    if solution.iter().any(|value| !value.is_finite()) {
        return Err(VelocityFourierSpongeError::NonFiniteTransform);
    }
    Ok(solution)
}

impl VlasovPoisson1d1v {
    /// Apply high-|s| damping, then minimally correct every x-row to restore
    /// its pre-sponge discrete mass, momentum and second velocity moment.
    ///
    /// The correction is the minimum L2 correction in span {1,v,v^2}.
    /// A candidate below the caller-declared positivity floor is rejected
    /// without mutating the original solver.
    pub fn apply_moment_constrained_velocity_fourier_sponge(
        &mut self,
        duration: f64,
        start_fraction_of_nyquist: f64,
        maximum_damping_rate: f64,
        exponent: u32,
        positivity_floor: f64,
    ) -> Result<VelocityFourierSpongeReport, VelocityFourierSpongeError> {
        if !positivity_floor.is_finite() || positivity_floor < 0.0 {
            return Err(VelocityFourierSpongeError::InvalidPositivityFloor);
        }

        let original_distribution = self.distribution.clone();
        let mut candidate_solver = self.clone();
        let mut report = candidate_solver.apply_velocity_fourier_sponge(
            duration,
            start_fraction_of_nyquist,
            maximum_damping_rate,
            exponent,
        )?;
        let config = self.config;

        // Scale velocity to [-1,1] before forming the polynomial Gram
        // matrix. This leaves the correction span unchanged but improves
        // conditioning when the physical velocity extent is enlarged.
        let velocity_scale = config
            .velocity_min
            .abs()
            .max(config.velocity_max.abs())
            .max(1.0);
        let mut gram = [[0.0; 3]; 3];
        for iv in 0..config.nv {
            let normalized_velocity = config.velocity(iv) / velocity_scale;
            let basis = [
                1.0,
                normalized_velocity,
                normalized_velocity * normalized_velocity,
            ];
            for row in 0..3 {
                for column in 0..3 {
                    gram[row][column] += config.dv() * basis[row] * basis[column];
                }
            }
        }

        let mut correction_l2_squared = 0.0;
        let mut maximum_pointwise_correction = 0.0_f64;
        let mut max_absolute_moment_residuals = [0.0_f64; 3];
        let mut max_relative_moment_residuals = [0.0_f64; 3];

        for ix in 0..config.nx {
            let start = ix * config.nv;
            let end = start + config.nv;
            let target_row = &original_distribution[start..end];
            let candidate_row = &mut candidate_solver.distribution[start..end];
            let target_moments = discrete_velocity_moments(target_row, config);
            // Constraints are solved in the scaled basis {1,u,u^2};
            // this is algebraically equivalent to {1,v,v^2}.
            let mut target_scaled_moments = [0.0; 3];
            let mut filtered_scaled_moments = [0.0; 3];
            for (iv, (target, filtered)) in target_row.iter().zip(candidate_row.iter()).enumerate()
            {
                let u = config.velocity(iv) / velocity_scale;
                let weight = config.dv();
                target_scaled_moments[0] += target * weight;
                target_scaled_moments[1] = u.mul_add(target * weight, target_scaled_moments[1]);
                target_scaled_moments[2] =
                    (u * u).mul_add(target * weight, target_scaled_moments[2]);
                filtered_scaled_moments[0] += filtered * weight;
                filtered_scaled_moments[1] =
                    u.mul_add(filtered * weight, filtered_scaled_moments[1]);
                filtered_scaled_moments[2] =
                    (u * u).mul_add(filtered * weight, filtered_scaled_moments[2]);
            }
            let residual = [
                target_scaled_moments[0] - filtered_scaled_moments[0],
                target_scaled_moments[1] - filtered_scaled_moments[1],
                target_scaled_moments[2] - filtered_scaled_moments[2],
            ];
            let correction = solve_three_by_three(gram, residual)?;

            for (iv, value) in candidate_row.iter_mut().enumerate() {
                let normalized_velocity = config.velocity(iv) / velocity_scale;
                let delta = correction[0]
                    + correction[1] * normalized_velocity
                    + correction[2] * normalized_velocity * normalized_velocity;
                *value += delta;
                correction_l2_squared =
                    (delta * delta).mul_add(config.dx() * config.dv(), correction_l2_squared);
                maximum_pointwise_correction = maximum_pointwise_correction.max(delta.abs());
                if !value.is_finite() {
                    return Err(VelocityFourierSpongeError::NonFiniteTransform);
                }
            }

            let corrected_moments = discrete_velocity_moments(candidate_row, config);
            for moment in 0..3 {
                let absolute = (corrected_moments[moment] - target_moments[moment]).abs();
                let scale = target_row
                    .iter()
                    .enumerate()
                    .map(|(iv, value)| {
                        let velocity = config.velocity(iv).abs();
                        config.dv()
                            * value.abs()
                            * match moment {
                                0 => 1.0,
                                1 => velocity,
                                _ => velocity * velocity,
                            }
                    })
                    .sum::<f64>()
                    .max(target_moments[moment].abs())
                    .max(f64::MIN_POSITIVE);
                max_absolute_moment_residuals[moment] =
                    max_absolute_moment_residuals[moment].max(absolute);
                let relative_residual = absolute / scale;
                max_relative_moment_residuals[moment] =
                    max_relative_moment_residuals[moment].max(relative_residual);
                if relative_residual > MOMENT_PROJECTION_RELATIVE_TOLERANCE {
                    return Err(VelocityFourierSpongeError::MomentResidualExceeded {
                        moment,
                        relative_residual,
                        tolerance: MOMENT_PROJECTION_RELATIVE_TOLERANCE,
                    });
                }
            }
        }

        let minimum_distribution_after = candidate_solver
            .distribution
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if minimum_distribution_after < positivity_floor {
            return Err(VelocityFourierSpongeError::PositivityFloorViolation {
                minimum: minimum_distribution_after,
                floor: positivity_floor,
            });
        }

        candidate_solver
            .refresh_electric_field()
            .and_then(|()| candidate_solver.ensure_finite_state())
            .map_err(VelocityFourierSpongeError::Solver)?;

        let after = candidate_solver
            .quadratic_casimir_budget()
            .map_err(VelocityFourierSpongeError::Solver)?;
        let diagnostics_after = candidate_solver
            .diagnostics()
            .map_err(VelocityFourierSpongeError::Solver)?;

        // Recompute the complete spectral C2 change after the polynomial
        // projection, including its cross terms and restored moments.
        let mut before_line = vec![Complex::ZERO; config.nv];
        let mut after_line = vec![Complex::ZERO; config.nv];
        let spectral_weight = config.dx() * config.dv() / config.nv as f64;
        let mut spectral_removed_casimir = 0.0;
        let mut high_s_casimir_before = 0.0;
        let mut high_s_casimir_after = 0.0;
        let mut low_s_casimir_before = 0.0;
        let mut low_s_casimir_after = 0.0;
        let nyquist_mode = config.nv / 2;
        for ix in 0..config.nx {
            for iv in 0..config.nv {
                let index = ix * config.nv + iv;
                before_line[iv] = Complex::new(original_distribution[index], 0.0);
                after_line[iv] = Complex::new(candidate_solver.distribution[index], 0.0);
            }
            fft(&mut before_line, false);
            fft(&mut after_line, false);
            for mode in 0..config.nv {
                let before_power = before_line[mode].re.mul_add(
                    before_line[mode].re,
                    before_line[mode].im * before_line[mode].im,
                );
                let after_power = after_line[mode].re.mul_add(
                    after_line[mode].re,
                    after_line[mode].im * after_line[mode].im,
                );
                spectral_removed_casimir += (before_power - after_power) * spectral_weight;
                let normalized_mode =
                    signed_mode(mode, config.nv).unsigned_abs() as f64 / nyquist_mode as f64;
                if normalized_mode > report.start_fraction_of_nyquist {
                    high_s_casimir_before += before_power * spectral_weight;
                    high_s_casimir_after += after_power * spectral_weight;
                } else {
                    low_s_casimir_before += before_power * spectral_weight;
                    low_s_casimir_after += after_power * spectral_weight;
                }
            }
        }

        let removed_total_casimir = report.casimir_before.total_casimir - after.total_casimir;
        let removed_spatial_mean_casimir =
            report.casimir_before.spatial_mean_casimir - after.spatial_mean_casimir;
        let removed_fluctuation_f_strophy =
            report.casimir_before.fluctuation_f_strophy - after.fluctuation_f_strophy;
        let casimir_accounting_residual = removed_total_casimir - spectral_removed_casimir;

        let all_finite = [
            correction_l2_squared,
            maximum_pointwise_correction,
            spectral_removed_casimir,
            removed_total_casimir,
            removed_spatial_mean_casimir,
            removed_fluctuation_f_strophy,
            casimir_accounting_residual,
            high_s_casimir_before,
            high_s_casimir_after,
            low_s_casimir_before,
            low_s_casimir_after,
        ]
        .iter()
        .all(|value| value.is_finite())
            && max_absolute_moment_residuals
                .iter()
                .chain(max_relative_moment_residuals.iter())
                .all(|value| value.is_finite());
        if !all_finite {
            return Err(VelocityFourierSpongeError::NonFiniteTransform);
        }

        report.casimir_after = after;
        report.diagnostics_after = diagnostics_after;
        report.removed_total_casimir = removed_total_casimir;
        report.removed_spatial_mean_casimir = removed_spatial_mean_casimir;
        report.removed_fluctuation_f_strophy = removed_fluctuation_f_strophy;
        report.spectral_removed_casimir = spectral_removed_casimir;
        report.casimir_accounting_residual = casimir_accounting_residual;
        report.moment_correction_l2 = correction_l2_squared.sqrt();
        report.maximum_pointwise_moment_correction = maximum_pointwise_correction;
        report.max_absolute_moment_residuals = max_absolute_moment_residuals;
        report.max_relative_moment_residuals = max_relative_moment_residuals;
        report.minimum_distribution_after = minimum_distribution_after;
        report.negative_value_fraction_before = negative_value_fraction(&original_distribution);
        report.negative_value_fraction_after =
            negative_value_fraction(&candidate_solver.distribution);
        report.high_s_casimir_before = high_s_casimir_before;
        report.high_s_casimir_after = high_s_casimir_after;
        report.high_s_casimir_removed = high_s_casimir_before - high_s_casimir_after;
        report.low_s_casimir_before = low_s_casimir_before;
        report.low_s_casimir_after = low_s_casimir_after;
        report.low_s_casimir_change = low_s_casimir_after - low_s_casimir_before;
        report.preserves_first_three_moments = true;

        *self = candidate_solver;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VlasovPoissonConfig;
    use std::f64::consts::TAU;

    fn config() -> VlasovPoissonConfig {
        VlasovPoissonConfig {
            nx: 64,
            nv: 256,
            domain_length: 4.0 * std::f64::consts::PI,
            velocity_min: -6.0,
            velocity_max: 6.0,
            charge_to_mass: -1.0,
            permittivity: 1.0,
            background_density: 1.0,
        }
    }

    fn distribution(with_x_dependence: bool) -> Vec<f64> {
        let config = config();
        let mut values = vec![0.0; config.nx * config.nv];
        for ix in 0..config.nx {
            let spatial = if with_x_dependence {
                1.0 + 0.05 * (TAU * config.x(ix) / config.domain_length).cos()
            } else {
                1.0
            };
            for iv in 0..config.nv {
                let phase = TAU * 8.0 * iv as f64 / config.nv as f64;
                let high_phase = TAU * 112.0 * iv as f64 / config.nv as f64;
                values[ix * config.nv + iv] =
                    spatial * (1.0 + 0.1 * phase.cos() + 0.2 * high_phase.cos());
            }
        }
        values
    }

    fn cosine_amplitude(distribution: &[f64], mode: usize) -> f64 {
        let config = config();
        let mut projection = 0.0;
        for ix in 0..config.nx {
            for iv in 0..config.nv {
                let phase = TAU * mode as f64 * iv as f64 / config.nv as f64;
                projection += distribution[ix * config.nv + iv] * phase.cos();
            }
        }
        2.0 * projection / (config.nx * config.nv) as f64
    }

    #[test]
    fn sponge_damps_high_s_and_preserves_low_s_density_and_clock() {
        let config = config();
        let initial = distribution(false);
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        let time_before = solver.time();
        let density_mode_before = solver.electric_mode_coefficient(1).unwrap();
        let expected_high = 0.2 * (-0.5625_f64).exp();

        let report = solver
            .apply_velocity_fourier_sponge(0.5, 0.5, 2.0, 2)
            .unwrap();
        let high_after = cosine_amplitude(solver.distribution(), 112);
        let low_after = cosine_amplitude(solver.distribution(), 8);
        let low_before = cosine_amplitude(&initial, 8);
        let density_mode_after = solver.electric_mode_coefficient(1).unwrap();

        assert!((high_after - expected_high).abs() < 1.0e-10);
        assert!((low_after - low_before).abs() < 1.0e-10);
        assert!((density_mode_after.amplitude() - density_mode_before.amplitude()).abs() < 1.0e-12);
        let mass_delta =
            (report.diagnostics_after.total_mass - report.diagnostics_before.total_mass).abs();
        let mass_scale = report.diagnostics_before.total_mass.abs().max(1.0);
        assert!(
            mass_delta < 1.0e-12 * mass_scale,
            "zero-velocity-mode mass drift = {mass_delta:e} (scale {mass_scale:e})"
        );
        assert!(report.diagnostics_after.minimum_distribution > 0.0);
        assert_eq!(
            report.diagnostics_after.time,
            report.diagnostics_before.time
        );
        assert_eq!(solver.time(), time_before);
        assert!((report.nyquist_attenuation - (-1.0_f64).exp()).abs() < 1.0e-14);
        assert!(report.removed_total_casimir > 0.0);
    }

    #[test]
    fn sponge_reports_parseval_casimir_and_fluctuation_loss() {
        let mut solver = VlasovPoisson1d1v::new(config(), distribution(true)).unwrap();
        let report = solver
            .apply_velocity_fourier_sponge(0.5, 0.5, 2.0, 2)
            .unwrap();
        let scale = report.casimir_before.total_casimir.max(1.0);
        assert!(report.removed_total_casimir > 0.0);
        assert!(report.removed_fluctuation_f_strophy > 0.0);
        assert!(
            report.casimir_accounting_residual.abs() < 1.0e-11 * scale,
            "C2 accounting residual = {:e}",
            report.casimir_accounting_residual
        );
        assert!(
            (report.removed_total_casimir
                - report.removed_spatial_mean_casimir
                - report.removed_fluctuation_f_strophy)
                .abs()
                < 1.0e-11 * scale
        );
    }

    #[test]
    fn moment_constrained_sponge_preserves_each_row_mass_momentum_and_energy() {
        let config = config();
        let initial = distribution(true);
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();

        let report = solver
            .apply_moment_constrained_velocity_fourier_sponge(0.5, 0.5, 2.0, 2, 0.0)
            .unwrap();

        for ix in 0..config.nx {
            let start = ix * config.nv;
            let end = start + config.nv;
            let before = discrete_velocity_moments(&initial[start..end], config);
            let after = discrete_velocity_moments(&solver.distribution()[start..end], config);
            for moment in 0..3 {
                assert!(
                    (before[moment] - after[moment]).abs()
                        <= 1.0e-11 * before[moment].abs().max(1.0),
                    "row={ix}, moment={moment}, before={}, after={}",
                    before[moment],
                    after[moment]
                );
            }
        }

        assert!(report.preserves_first_three_moments);
        assert!(report.moment_correction_l2 > 0.0);
        assert!(report.maximum_pointwise_moment_correction > 0.0);
        assert!(report.minimum_distribution_after >= 0.0);
        assert_eq!(report.negative_value_fraction_after, 0.0);
        assert!(
            report
                .max_relative_moment_residuals
                .iter()
                .all(|value| *value <= MOMENT_PROJECTION_RELATIVE_TOLERANCE)
        );
        assert!(report.high_s_casimir_before > report.high_s_casimir_after);
        assert!(report.high_s_casimir_removed > 0.0);
        assert!(report.low_s_casimir_before > 0.0);
        assert!(report.removed_total_casimir > 0.0);
        assert!(
            report.casimir_accounting_residual.abs()
                < 1.0e-11 * report.casimir_before.total_casimir.max(1.0)
        );
        assert!(
            (report.diagnostics_after.total_mass - report.diagnostics_before.total_mass).abs()
                < 1.0e-11
        );
        assert!(
            (report.diagnostics_after.momentum - report.diagnostics_before.momentum).abs()
                < 1.0e-11
        );
        assert!(
            (report.diagnostics_after.kinetic_energy - report.diagnostics_before.kinetic_energy)
                .abs()
                < 1.0e-10
        );
    }

    #[test]
    fn moment_constrained_sponge_is_well_conditioned_on_expanded_velocity_support() {
        let config = VlasovPoissonConfig {
            nx: 64,
            nv: 256,
            domain_length: 4.0 * std::f64::consts::PI,
            velocity_min: -60.0,
            velocity_max: 60.0,
            charge_to_mass: -1.0,
            permittivity: 1.0,
            background_density: 1.0,
        };
        let mut initial = vec![0.0; config.nx * config.nv];
        for ix in 0..config.nx {
            let spatial = 1.0 + 0.01 * (TAU * config.x(ix) / config.domain_length).cos();
            for iv in 0..config.nv {
                let phase = TAU * iv as f64 / config.nv as f64;
                initial[ix * config.nv + iv] = spatial
                    * (1.0 + 0.05 * (8.0 * phase).cos() + 0.1 * (112.0 * phase).cos());
            }
        }
        assert!(initial.iter().all(|value| *value > 0.8));
        let mut solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        let report = solver
            .apply_moment_constrained_velocity_fourier_sponge(0.1, 0.5, 0.25, 2, 0.0)
            .unwrap();
        assert!(report.preserves_first_three_moments);
        assert!(
            report
                .max_relative_moment_residuals
                .iter()
                .all(|value| *value <= MOMENT_PROJECTION_RELATIVE_TOLERANCE)
        );
        assert!(report.casimir_accounting_residual.abs() < 1.0e-8);
    }

    #[test]
    fn zero_strength_moment_constrained_sponge_is_exact_identity() {
        let config = config();
        let initial = distribution(true);
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        let report = solver
            .apply_moment_constrained_velocity_fourier_sponge(0.0, 0.5, 2.0, 2, 0.0)
            .unwrap();
        assert_eq!(solver.distribution(), initial.as_slice());
        assert!(report.preserves_first_three_moments);
        assert_eq!(report.moment_correction_l2, 0.0);
        assert_eq!(report.removed_total_casimir, 0.0);
        assert_eq!(report.high_s_casimir_removed, 0.0);
    }

    #[test]
    fn moment_constrained_sponge_rejects_positivity_floor_without_mutation() {
        let config = config();
        let initial = distribution(true);
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        let time_before = solver.time();

        let result = solver.apply_moment_constrained_velocity_fourier_sponge(0.5, 0.5, 2.0, 2, 2.0);
        assert!(matches!(
            result,
            Err(VelocityFourierSpongeError::PositivityFloorViolation { .. })
        ));
        assert_eq!(solver.distribution(), initial.as_slice());
        assert_eq!(solver.time(), time_before);
    }

    #[test]
    fn invalid_moment_sponge_floor_fails_without_mutation() {
        let config = config();
        let initial = distribution(true);
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        for floor in [f64::NAN, f64::INFINITY, -1.0] {
            assert_eq!(
                solver.apply_moment_constrained_velocity_fourier_sponge(0.5, 0.5, 2.0, 2, floor),
                Err(VelocityFourierSpongeError::InvalidPositivityFloor)
            );
            assert_eq!(solver.distribution(), initial.as_slice());
        }
    }

    #[test]
    fn zero_duration_sponge_is_an_identity() {
        let initial = distribution(true);
        let mut solver = VlasovPoisson1d1v::new(config(), initial.clone()).unwrap();
        let report = solver
            .apply_velocity_fourier_sponge(0.0, 0.5, 2.0, 2)
            .unwrap();
        assert_eq!(solver.distribution(), initial.as_slice());
        assert_eq!(report.removed_total_casimir, 0.0);
    }

    #[test]
    fn invalid_sponge_parameters_fail_without_mutation() {
        let initial = distribution(true);
        let mut solver = VlasovPoisson1d1v::new(config(), initial.clone()).unwrap();
        let invalid = [
            (f64::NAN, 0.5, 2.0, 2),
            (-1.0, 0.5, 2.0, 2),
            (0.5, f64::NAN, 2.0, 2),
            (0.5, 1.0, 2.0, 2),
            (0.5, 0.5, f64::INFINITY, 2),
            (0.5, 0.5, -1.0, 2),
            (0.5, 0.5, 2.0, 0),
            (0.5, 0.5, 2.0, 17),
        ];
        for (duration, start, damping, exponent) in invalid {
            assert!(
                solver
                    .apply_velocity_fourier_sponge(duration, start, damping, exponent)
                    .is_err()
            );
            assert_eq!(solver.distribution(), initial.as_slice());
            assert_eq!(solver.time(), 0.0);
        }
    }
}
