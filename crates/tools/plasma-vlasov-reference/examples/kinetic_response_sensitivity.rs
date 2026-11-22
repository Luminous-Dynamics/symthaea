use std::error::Error;
use std::f64::consts::TAU;

use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

const DT: f64 = 0.025;
const FIRST_PERTURBATION: f64 = 1.0e-2;
const FIRST_MODE: usize = 1;
const SECOND_MODE: usize = 2;
const SECOND_PULSE_TIME: f64 = 2.0;
const SECOND_PULSE_AMPLITUDE: f64 = 5.0e-2;
const FUTURE_HORIZON: f64 = 5.2;
const REQUESTED_EPSILON: f64 = 1.0e-3;
const POSITIVITY_FLOOR: f64 = 1.0e-12;
const VELOCITY_MODES: [usize; 8] = [4, 8, 16, 32, 48, 64, 80, 112];
const VELOCITY_PHASES: [f64; 2] = [0.0, std::f64::consts::FRAC_PI_2];

type ComplexTraceSample = (f64, f64, f64);
type ComplexTrace = Vec<ComplexTraceSample>;
type SensitivityTracePair = (ComplexTrace, ComplexTrace);

#[derive(Debug, Clone, Copy)]
struct ResponsePeak {
    time: f64,
    magnitude: f64,
    phase: f64,
    fwhm: f64,
}

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

fn solve_three_by_three(gram: [[f64; 3]; 3], rhs: [f64; 3]) -> Result<[f64; 3], Box<dyn Error>> {
    let mut matrix = [[0.0; 4]; 3];
    for row in 0..3 {
        for column in 0..3 {
            matrix[row][column] = gram[row][column];
        }
        matrix[row][3] = rhs[row];
    }
    for column in 0..3 {
        let pivot_row = (column..3)
            .max_by(|left, right| {
                matrix[*left][column]
                    .abs()
                    .total_cmp(&matrix[*right][column].abs())
            })
            .ok_or("missing pivot")?;
        let pivot = matrix[pivot_row][column];
        if !pivot.is_finite() || pivot.abs() < 1.0e-13 {
            return Err("singular moment projection".into());
        }
        matrix.swap(pivot_row, column);
        let diagonal = matrix[column][column];
        for value in &mut matrix[column][column..4] {
            *value /= diagonal;
        }
        let pivot_values = matrix[column];
        for (row_index, row_values) in matrix.iter_mut().enumerate() {
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
    let solution = [matrix[0][3], matrix[1][3], matrix[2][3]];
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("non-finite moment projection".into());
    }
    Ok(solution)
}

/// Build a real perturbation in the per-x nullspace of [1,v,v^2].
fn moment_neutral_direction(
    base: &[f64],
    config: VlasovPoissonConfig,
    spatial_mode: usize,
    velocity_mode: usize,
    velocity_phase: f64,
) -> Result<(Vec<f64>, f64), Box<dyn Error>> {
    let velocity_scale = config
        .velocity_min
        .abs()
        .max(config.velocity_max.abs())
        .max(1.0);
    let mut gram = [[0.0; 3]; 3];
    for iv in 0..config.nv {
        let u = config.velocity(iv) / velocity_scale;
        let basis = [1.0, u, u * u];
        for row in 0..3 {
            for column in 0..3 {
                gram[row][column] += config.dv() * basis[row] * basis[column];
            }
        }
    }

    let mut direction = vec![0.0; base.len()];
    let mut maximum_relative_moment_residual = 0.0_f64;
    let spatial_wave_number = TAU * spatial_mode as f64 / config.domain_length;
    let velocity_length = config.velocity_max - config.velocity_min;

    for ix in 0..config.nx {
        let start = ix * config.nv;
        let x_factor = (spatial_wave_number * config.x(ix)).cos();
        let mut raw = vec![0.0; config.nv];
        let mut moments = [0.0; 3];

        for iv in 0..config.nv {
            let velocity = config.velocity(iv);
            let u = velocity / velocity_scale;
            let angle = TAU * (velocity - config.velocity_min) / velocity_length;
            let value = base[start + iv]
                * x_factor
                * (velocity_mode as f64 * angle + velocity_phase).cos();
            raw[iv] = value;
            let weighted = value * config.dv();
            moments[0] += weighted;
            moments[1] = u.mul_add(weighted, moments[1]);
            moments[2] = (u * u).mul_add(weighted, moments[2]);
        }

        let coefficients = solve_three_by_three(gram, moments)?;
        for iv in 0..config.nv {
            let u = config.velocity(iv) / velocity_scale;
            let polynomial = coefficients[0] + coefficients[1] * u + coefficients[2] * u * u;
            direction[start + iv] = raw[iv] - polynomial;
        }

        let mut residual = [0.0; 3];
        let mut scales = [0.0; 3];
        for iv in 0..config.nv {
            let u = config.velocity(iv) / velocity_scale;
            let value = direction[start + iv];
            let weighted = value * config.dv();
            residual[0] += weighted;
            residual[1] = u.mul_add(weighted, residual[1]);
            residual[2] = (u * u).mul_add(weighted, residual[2]);
            let absolute = value.abs() * config.dv();
            scales[0] += absolute;
            scales[1] += absolute * u.abs();
            scales[2] += absolute * u * u;
        }
        for moment in 0..3 {
            let relative = residual[moment].abs() / scales[moment].max(f64::MIN_POSITIVE);
            maximum_relative_moment_residual =
                maximum_relative_moment_residual.max(relative);
        }
    }

    if maximum_relative_moment_residual > 1.0e-11 {
        return Err("moment-neutral projection residual exceeds tolerance".into());
    }
    Ok((direction, maximum_relative_moment_residual))
}

fn positivity_limited_epsilon(
    pulse_base: &[f64],
    control_base: &[f64],
    direction: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let mut maximum_epsilon = f64::INFINITY;
    for ((pulse, control), delta) in pulse_base.iter().zip(control_base).zip(direction) {
        if delta.abs() > f64::MIN_POSITIVE {
            maximum_epsilon = maximum_epsilon.min(
                (pulse.min(control) - POSITIVITY_FLOOR) / delta.abs(),
            );
        }
    }
    let epsilon = REQUESTED_EPSILON.min(0.25 * maximum_epsilon);
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("no positive symmetric perturbation satisfies the positivity floor".into());
    }
    Ok(epsilon)
}

fn interpolate_crossing(
    left_time: f64,
    left_value: f64,
    right_time: f64,
    right_value: f64,
    target: f64,
) -> f64 {
    let difference = right_value - left_value;
    if difference.abs() <= f64::MIN_POSITIVE {
        return 0.5 * (left_time + right_time);
    }
    left_time + (target - left_value) * (right_time - left_time) / difference
}

fn peak(samples: &[ComplexTraceSample]) -> ResponsePeak {
    let amplitudes: Vec<f64> = samples
        .iter()
        .map(|(_, real, imaginary)| real.hypot(*imaginary))
        .collect();
    let peak_index = amplitudes
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let magnitude = amplitudes.get(peak_index).copied().unwrap_or(0.0);
    let half = 0.5 * magnitude;

    let mut left_crossing = None;
    for index in (1..=peak_index).rev() {
        if amplitudes[index - 1] <= half && amplitudes[index] >= half {
            left_crossing = Some(interpolate_crossing(
                samples[index - 1].0,
                amplitudes[index - 1],
                samples[index].0,
                amplitudes[index],
                half,
            ));
            break;
        }
    }
    let mut right_crossing = None;
    for index in peak_index..samples.len().saturating_sub(1) {
        if amplitudes[index] >= half && amplitudes[index + 1] <= half {
            right_crossing = Some(interpolate_crossing(
                samples[index].0,
                amplitudes[index],
                samples[index + 1].0,
                amplitudes[index + 1],
                half,
            ));
            break;
        }
    }
    let fwhm = match (left_crossing, right_crossing) {
        (Some(left), Some(right)) => right - left,
        _ => f64::NAN,
    };
    let (_, real, imaginary) = samples.get(peak_index).copied().unwrap_or((0.0, 0.0, 0.0));
    ResponsePeak {
        time: samples.get(peak_index).map(|sample| sample.0).unwrap_or(0.0),
        magnitude,
        phase: imaginary.atan2(real),
        fwhm,
    }
}

fn central_difference_trace(
    config: VlasovPoissonConfig,
    base: &[f64],
    direction: &[f64],
    epsilon: f64,
    future_steps: usize,
    sample_every: usize,
) -> Result<SensitivityTracePair, Box<dyn Error>> {
    let plus_distribution: Vec<f64> = base
        .iter()
        .zip(direction)
        .map(|(value, delta)| value + epsilon * delta)
        .collect();
    let minus_distribution: Vec<f64> = base
        .iter()
        .zip(direction)
        .map(|(value, delta)| value - epsilon * delta)
        .collect();

    if plus_distribution
        .iter()
        .chain(&minus_distribution)
        .any(|value| !value.is_finite() || *value < POSITIVITY_FLOOR)
    {
        return Err("symmetric perturbation violates positivity floor".into());
    }

    let mut plus = VlasovPoisson1d1v::new(config, plus_distribution)?;
    let mut minus = VlasovPoisson1d1v::new(config, minus_distribution)?;
    let mut target_sensitivity = Vec::new();
    let mut off_target_sensitivity = Vec::new();

    for step in 0..=future_steps {
        if step % sample_every == 0 || step == future_steps {
            let plus_target = plus.electric_mode_coefficient(FIRST_MODE)?;
            let minus_target = minus.electric_mode_coefficient(FIRST_MODE)?;
            let plus_off = plus.electric_mode_coefficient(SECOND_MODE)?;
            let minus_off = minus.electric_mode_coefficient(SECOND_MODE)?;

            let inverse = 0.5 / epsilon;
            target_sensitivity.push((
                step as f64 * DT,
                (plus_target.real - minus_target.real) * inverse,
                (plus_target.imaginary - minus_target.imaginary) * inverse,
            ));
            off_target_sensitivity.push((
                step as f64 * DT,
                (plus_off.real - minus_off.real) * inverse,
                (plus_off.imaginary - minus_off.imaginary) * inverse,
            ));
        }
        if step < future_steps {
            plus.step(DT)?;
            minus.step(DT)?;
        }
    }

    Ok((target_sensitivity, off_target_sensitivity))
}

fn subtract_traces(left: &[ComplexTraceSample], right: &[ComplexTraceSample]) -> ComplexTrace {
    left.iter()
        .zip(right)
        .map(|(pulse, control)| {
            (
                pulse.0,
                pulse.1 - control.1,
                pulse.2 - control.2,
            )
        })
        .collect()
}

fn sensitivity_at_time(
    samples: &[ComplexTraceSample],
    target_time: f64,
) -> ComplexTraceSample {
    samples
        .iter()
        .copied()
        .min_by(|left, right| {
            (left.0 - target_time)
                .abs()
                .total_cmp(&(right.0 - target_time).abs())
        })
        .unwrap_or((0.0, 0.0, 0.0))
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = config();
    let initial = maxwellian_perturbation(config, FIRST_PERTURBATION, FIRST_MODE, 1.0)?;
    let mut checkpoint_solver = VlasovPoisson1d1v::new(config, initial)?;
    let pulse_steps = (SECOND_PULSE_TIME / DT).round() as usize;
    let future_steps = (FUTURE_HORIZON / DT).round() as usize;
    if (pulse_steps as f64 * DT - SECOND_PULSE_TIME).abs() > 1.0e-12
        || (future_steps as f64 * DT - FUTURE_HORIZON).abs() > 1.0e-12
    {
        return Err("pulse and horizon must lie on the timestep grid".into());
    }

    for _ in 0..pulse_steps {
        checkpoint_solver.step(DT)?;
    }
    let control_checkpoint = checkpoint_solver.distribution().to_vec();
    let second_kick: Vec<f64> = (0..config.nx)
        .map(|ix| {
            SECOND_PULSE_AMPLITUDE
                * (TAU * SECOND_MODE as f64 * config.x(ix) / config.domain_length).sin()
        })
        .collect();
    checkpoint_solver.apply_velocity_impulse(&second_kick)?;
    let pulse_checkpoint = checkpoint_solver.distribution().to_vec();

    println!(
        "spatial_mode,velocity_mode,velocity_phase,epsilon,minimum_pulse_plus,minimum_pulse_minus,minimum_control_plus,minimum_control_minus,max_relative_moment_residual,ballistic_rephase_elapsed,target_pulse_dE_rephase_magnitude,target_pulse_dE_rephase_phase,target_control_dE_rephase_magnitude,target_intervention_dE_rephase_magnitude,target_intervention_dE_rephase_phase,intervention_susceptibility_peak_time,intervention_susceptibility_peak,intervention_susceptibility_phase,intervention_susceptibility_fwhm,intervention_mode_peak_sensitivity,intervention_to_echo_sensitivity_ratio,epsilon_half_peak_relative_delta"
    );

    for velocity_mode in VELOCITY_MODES {
        for velocity_phase in VELOCITY_PHASES {
            let (direction, moment_residual) = moment_neutral_direction(
                &pulse_checkpoint,
                config,
                FIRST_MODE,
                velocity_mode,
                velocity_phase,
            )?;
            let epsilon = positivity_limited_epsilon(
                &pulse_checkpoint,
                &control_checkpoint,
                &direction,
            )?;

            let (pulse_trace, pulse_off_trace) = central_difference_trace(
                config,
                &pulse_checkpoint,
                &direction,
                epsilon,
                future_steps,
                2,
            )?;
            let (control_trace, control_off_trace) = central_difference_trace(
                config,
                &control_checkpoint,
                &direction,
                epsilon,
                future_steps,
                2,
            )?;
            let intervention_trace = subtract_traces(&pulse_trace, &control_trace);
            let intervention_off_trace =
                subtract_traces(&pulse_off_trace, &control_off_trace);
            let intervention_peak = peak(&intervention_trace);
            let intervention_mode_peak = peak(&intervention_off_trace);
            let pulse_rephase = sensitivity_at_time(&pulse_trace, 2.0);
            let control_rephase = sensitivity_at_time(&control_trace, 2.0);
            let intervention_rephase = sensitivity_at_time(&intervention_trace, 2.0);

            let half_epsilon = 0.5 * epsilon;
            let (half_pulse_trace, _) = central_difference_trace(
                config,
                &pulse_checkpoint,
                &direction,
                half_epsilon,
                future_steps,
                2,
            )?;
            let (half_control_trace, _) = central_difference_trace(
                config,
                &control_checkpoint,
                &direction,
                half_epsilon,
                future_steps,
                2,
            )?;
            let half_intervention_trace =
                subtract_traces(&half_pulse_trace, &half_control_trace);
            let half_intervention_peak = peak(&half_intervention_trace);
            let half_relative_delta = (half_intervention_peak.magnitude
                - intervention_peak.magnitude)
                .abs()
                / intervention_peak.magnitude.max(f64::MIN_POSITIVE);
            let off_target_ratio = intervention_mode_peak.magnitude
                / intervention_peak.magnitude.max(f64::MIN_POSITIVE);

            let minima = [
                pulse_checkpoint
                    .iter()
                    .zip(&direction)
                    .map(|(value, delta)| value + epsilon * delta)
                    .fold(f64::INFINITY, f64::min),
                pulse_checkpoint
                    .iter()
                    .zip(&direction)
                    .map(|(value, delta)| value - epsilon * delta)
                    .fold(f64::INFINITY, f64::min),
                control_checkpoint
                    .iter()
                    .zip(&direction)
                    .map(|(value, delta)| value + epsilon * delta)
                    .fold(f64::INFINITY, f64::min),
                control_checkpoint
                    .iter()
                    .zip(&direction)
                    .map(|(value, delta)| value - epsilon * delta)
                    .fold(f64::INFINITY, f64::min),
            ];

            println!(
                "{},{},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},4.0,{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{:.12e},{:.12e},{:.8},{:.12e},{:.12e},{:.12e}",
                FIRST_MODE,
                velocity_mode,
                velocity_phase,
                epsilon,
                minima[0],
                minima[1],
                minima[2],
                minima[3],
                moment_residual,
                pulse_rephase.1.hypot(pulse_rephase.2),
                pulse_rephase.2.atan2(pulse_rephase.1),
                control_rephase.1.hypot(control_rephase.2),
                intervention_rephase.1.hypot(intervention_rephase.2),
                intervention_rephase.2.atan2(intervention_rephase.1),
                intervention_peak.time,
                intervention_peak.magnitude,
                intervention_peak.phase,
                intervention_peak.fwhm,
                intervention_mode_peak.magnitude,
                off_target_ratio,
                half_relative_delta,
            );
        }
    }

    Ok(())
}
