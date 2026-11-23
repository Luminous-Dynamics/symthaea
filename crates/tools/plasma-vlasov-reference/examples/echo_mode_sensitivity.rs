//! Finite-intervention estimate of future echo-response sensitivity to signed
//! phase-space modes at the pre-pulse state.
//!
//! This is a discrete paired-trajectory oracle, not an adjoint or a universal
//! kinetic-memory certificate. Each perturbation is projected to preserve the
//! local discrete density before the second pulse. The reported derivative is
//! the central finite difference of the baseline-subtracted complex E_k
//! response at a fixed future observation time.

use std::error::Error;
use std::f64::consts::TAU;

use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

const NX: usize = 32;
const NV: usize = 128;
const DT: f64 = 0.02;
const FIRST_MODE: usize = 1;
const SECOND_MODE: usize = 2;
const FIRST_TIME: f64 = 0.0;
const SECOND_TIME: f64 = 2.0;
const OBSERVATION_TIME: f64 = 7.2;
const INITIAL_AMPLITUDE: f64 = 1.0e-3;
const IMPULSE_AMPLITUDE: f64 = 5.0e-2;
const PERTURBATION_AMPLITUDE: f64 = 1.0e-3;
const SIGNED_VELOCITY_MODES: [isize; 12] = [-56, -40, -24, -16, -8, -4, 4, 8, 16, 24, 40, 56];

#[derive(Debug, Clone, Copy)]
struct ComplexResponse {
    real: f64,
    imaginary: f64,
}

impl ComplexResponse {
    fn amplitude(self) -> f64 {
        self.real.hypot(self.imaginary)
    }

    fn difference(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imaginary: self.imaginary - other.imaginary,
        }
    }

    fn midpoint(self, other: Self) -> Self {
        Self {
            real: 0.5 * (self.real + other.real),
            imaginary: 0.5 * (self.imaginary + other.imaginary),
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            real: self.real * factor,
            imaginary: self.imaginary * factor,
        }
    }
}

fn config() -> VlasovPoissonConfig {
    VlasovPoissonConfig {
        nx: NX,
        nv: NV,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -6.0,
        velocity_max: 6.0,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    }
}

/// Apply a relative cosine perturbation to one signed (k,s) phase-space mode,
/// subtracting its row-wise velocity-weighted mean so each x-cell's discrete
/// density is unchanged. The modulation remains positive for |epsilon|<0.5.
fn density_preserving_mode_perturbation(
    distribution: &[f64],
    config: VlasovPoissonConfig,
    signed_velocity_mode: isize,
    epsilon: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    if !epsilon.is_finite() || epsilon.abs() >= 0.5 {
        return Err("relative perturbation amplitude must be finite with magnitude < 0.5".into());
    }
    let expected_len = config.nx.checked_mul(config.nv).ok_or("grid size overflow")?;
    if distribution.len() != expected_len {
        return Err("distribution length does not match the configured grid".into());
    }
    let mut result = distribution.to_vec();
    let x_phase_scale = TAU * FIRST_MODE as f64 / config.domain_length;
    let velocity_phase_scale = TAU * signed_velocity_mode as f64 / config.nv as f64;

    for ix in 0..config.nx {
        let row_start = ix * config.nv;
        let row = &distribution[row_start..row_start + config.nv];
        let row_density: f64 = row.iter().sum();
        if !row_density.is_finite() || row_density <= 0.0 {
            return Err("pre-pulse row density must be finite and positive".into());
        }

        let weighted_cosine_mean = row
            .iter()
            .enumerate()
            .map(|(iv, value)| {
                let phase = x_phase_scale * config.x(ix) + velocity_phase_scale * iv as f64;
                value * phase.cos()
            })
            .sum::<f64>()
            / row_density;

        for (iv, value) in row.iter().copied().enumerate() {
            let phase = x_phase_scale * config.x(ix) + velocity_phase_scale * iv as f64;
            result[row_start + iv] =
                value * (1.0 + epsilon * (phase.cos() - weighted_cosine_mean));
        }
    }
    Ok(result)
}

fn density_change_max(
    before: &[f64],
    after: &[f64],
    config: VlasovPoissonConfig,
) -> f64 {
    let mut maximum = 0.0_f64;
    for ix in 0..config.nx {
        let start = ix * config.nv;
        let before_sum: f64 = before[start..start + config.nv].iter().sum();
        let after_sum: f64 = after[start..start + config.nv].iter().sum();
        maximum = maximum.max((after_sum - before_sum).abs());
    }
    maximum
}

fn mass_change_relative(
    before: &[f64],
    after: &[f64],
    config: VlasovPoissonConfig,
) -> f64 {
    let measure = config.dx() * config.dv();
    let before_mass = before.iter().sum::<f64>() * measure;
    let after_mass = after.iter().sum::<f64>() * measure;
    (after_mass - before_mass) / before_mass.abs().max(f64::MIN_POSITIVE)
}

fn response_for_initial_state(
    config: VlasovPoissonConfig,
    pre_pulse_distribution: &[f64],
    kick: &[f64],
    post_pulse_steps: usize,
) -> Result<ComplexResponse, Box<dyn Error>> {
    let mut pulse = VlasovPoisson1d1v::new(config, pre_pulse_distribution.to_vec())?;
    let mut control = VlasovPoisson1d1v::new(config, pre_pulse_distribution.to_vec())?;
    pulse.apply_velocity_impulse(kick)?;

    for _ in 0..post_pulse_steps {
        pulse.step(DT)?;
        control.step(DT)?;
    }

    let pulse_field = pulse.electric_mode_coefficient(FIRST_MODE)?;
    let control_field = control.electric_mode_coefficient(FIRST_MODE)?;
    Ok(ComplexResponse {
        real: pulse_field.real - control_field.real,
        imaginary: pulse_field.imaginary - control_field.imaginary,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = config();
    let pulse_step = (SECOND_TIME / DT).round() as usize;
    let observation_step = (OBSERVATION_TIME / DT).round() as usize;
    let post_pulse_steps = observation_step
        .checked_sub(pulse_step)
        .ok_or("observation time precedes the second pulse")?;

    if (pulse_step as f64 * DT - SECOND_TIME).abs() > 1.0e-12
        || (observation_step as f64 * DT - OBSERVATION_TIME).abs() > 1.0e-12
    {
        return Err("pulse and observation times must lie on the declared timestep grid".into());
    }

    let initial =
        maxwellian_perturbation(config, INITIAL_AMPLITUDE, FIRST_MODE, 1.0)?;
    let mut pre_pulse_solver = VlasovPoisson1d1v::new(config, initial)?;
    for _ in 0..pulse_step {
        pre_pulse_solver.step(DT)?;
    }
    let pre_pulse_distribution = pre_pulse_solver.distribution().to_vec();

    let kick: Vec<f64> = (0..config.nx)
        .map(|ix| {
            let phase = TAU * SECOND_MODE as f64 * config.x(ix) / config.domain_length;
            IMPULSE_AMPLITUDE * phase.sin()
        })
        .collect();

    let baseline_response = response_for_initial_state(
        config,
        &pre_pulse_distribution,
        &kick,
        post_pulse_steps,
    )?;
    let baseline_amplitude = baseline_response.amplitude();
    if !baseline_amplitude.is_finite() || baseline_amplitude <= f64::MIN_POSITIVE {
        return Err("baseline complex echo response is not finite and nonzero".into());
    }

    println!(
        "signed_velocity_mode,spatial_mode,observation_time,perturbation_epsilon,baseline_response_real,baseline_response_imag,baseline_response_amplitude,derivative_real,derivative_imag,derivative_amplitude,relative_sensitivity,epsilon_scaled_response_effect,derivative_phase_radians,even_order_residual_fraction,pre_pulse_density_change_max,perturbed_mass_change_relative"
    );

    for signed_mode in SIGNED_VELOCITY_MODES {
        let positive_distribution = density_preserving_mode_perturbation(
            &pre_pulse_distribution,
            config,
            signed_mode,
            PERTURBATION_AMPLITUDE,
        )?;
        let negative_distribution = density_preserving_mode_perturbation(
            &pre_pulse_distribution,
            config,
            signed_mode,
            -PERTURBATION_AMPLITUDE,
        )?;

        let positive_response = response_for_initial_state(
            config,
            &positive_distribution,
            &kick,
            post_pulse_steps,
        )?;
        let negative_response = response_for_initial_state(
            config,
            &negative_distribution,
            &kick,
            post_pulse_steps,
        )?;

        let derivative = positive_response
            .difference(negative_response)
            .scale(0.5 / PERTURBATION_AMPLITUDE);
        let midpoint = positive_response.midpoint(negative_response);
        let even_order_residual =
            midpoint.difference(baseline_response).amplitude() / baseline_amplitude;
        let relative_sensitivity = derivative.amplitude() / baseline_amplitude;
        let epsilon_scaled_response_effect =
            PERTURBATION_AMPLITUDE * relative_sensitivity;

        println!(
            "{signed_mode},{FIRST_MODE},{OBSERVATION_TIME:.8},{PERTURBATION_AMPLITUDE:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{:.12e},{:.12e},{:.12e}",
            baseline_response.real,
            baseline_response.imaginary,
            baseline_amplitude,
            derivative.real,
            derivative.imaginary,
            derivative.amplitude(),
            relative_sensitivity,
            epsilon_scaled_response_effect,
            derivative.imaginary.atan2(derivative.real),
            even_order_residual,
            density_change_max(&pre_pulse_distribution, &positive_distribution, config),
            mass_change_relative(&pre_pulse_distribution, &positive_distribution, config),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_mode_perturbations_preserve_each_discrete_spatial_density() {
        let config = config();
        let initial = maxwellian_perturbation(config, 1.0e-3, FIRST_MODE, 1.0).unwrap();
        let perturbed =
            density_preserving_mode_perturbation(&initial, config, 24, 1.0e-3).unwrap();
        assert!(
            density_change_max(&initial, &perturbed, config) < 1.0e-12,
            "row-wise density changed under the projected phase-space perturbation"
        );
        assert!(mass_change_relative(&initial, &perturbed, config).abs() < 1.0e-13);
        assert!(perturbed.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn signed_mode_perturbations_reject_invalid_amplitudes_and_shapes() {
        let config = config();
        let initial = maxwellian_perturbation(config, 1.0e-3, FIRST_MODE, 1.0).unwrap();
        assert!(
            density_preserving_mode_perturbation(&initial, config, 4, f64::NAN).is_err()
        );
        assert!(
            density_preserving_mode_perturbation(&initial, config, 4, 0.5).is_err()
        );
        assert!(
            density_preserving_mode_perturbation(&initial[..initial.len() - 1], config, 4, 1.0e-3)
                .is_err()
        );
    }
}
