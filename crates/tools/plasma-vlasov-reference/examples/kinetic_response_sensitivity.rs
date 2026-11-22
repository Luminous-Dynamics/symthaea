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

#[derive(Debug, Clone, Copy)]
struct ResponsePeak {
    time: f64,
    magnitude: f64,
    phase: f64,
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

/// Build a perturbation with zero per-row mass, momentum and second moment.
fn moment_neutral_direction(
    base: &[f64],
    config: VlasovPoissonConfig,
    spatial_mode: usize,
    velocity_mode: usize,
) -> Result<(Vec<f64>, f64, f64), Box<dyn Error>> {
    let mut gram = [[0.0; 3]; 3];
    for iv in 0..config.nv {
        let velocity = config.velocity(iv);
        let basis = [1.0, velocity, velocity * velocity];
        for row in 0..3 {
            for column in 0..3 {
                gram[row][column] += config.dv() * basis[row] * basis[column];
            }
        }
    }

    let mut direction = vec![0.0; base.len()];
    let mut maximum_moment_residual = 0.0_f64;
    let spatial_wave_number = TAU * spatial_mode as f64 / config.domain_length;
    let velocity_length = config.velocity_max - config.velocity_min;

    for ix in 0..config.nx {
        let start = ix * config.nv;
        let end = start + config.nv;
        let mut raw = vec![0.0; config.nv];
        let x_factor = (spatial_wave_number * config.x(ix)).cos();
        let mut moments = [0.0; 3];

        for iv in 0..config.nv {
            let velocity = config.velocity(iv);
            let angle = TAU * (velocity - config.velocity_min) / velocity_length;
            let value = base[start + iv] * x_factor * (velocity_mode as f64 * angle).cos();
            raw[iv] = value;
            let weighted = value * config.dv();
            moments[0] += weighted;
            moments[1] = velocity.mul_add(weighted, moments[1]);
            moments[2] = (velocity * velocity).mul_add(weighted, moments[2]);
        }

        let coefficients = solve_three_by_three(gram, moments)?;
        for iv in 0..config.nv {
            let velocity = config.velocity(iv);
            let polynomial = coefficients[0]
                + coefficients[1] * velocity
                + coefficients[2] * velocity * velocity;
            direction[start + iv] = raw[iv] - polynomial;
        }

        let mut residual = [0.0; 3];
        for iv in 0..config.nv {
            let velocity = config.velocity(iv);
            let weighted = direction[start + iv] * config.dv();
            residual[0] += weighted;
            residual[1] = velocity.mul_add(weighted, residual[1]);
            residual[2] = (velocity * velocity).mul_add(weighted, residual[2]);
        }
        for value in residual {
            maximum_moment_residual = maximum_moment_residual.max(value.abs());
        }
        let _ = end;
    }

    let mut maximum_epsilon = f64::INFINITY;
    for (value, delta) in base.iter().zip(&direction) {
        if delta.abs() > f64::MIN_POSITIVE {
            maximum_epsilon = maximum_epsilon.min((value - POSITIVITY_FLOOR) / delta.abs());
        }
    }
    let epsilon = REQUESTED_EPSILON.min(0.25 * maximum_epsilon);
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("no positive symmetric perturbation satisfies the positivity floor".into());
    }

    Ok((direction, epsilon, maximum_moment_residual))
}

type ComplexTraceSample = (f64, f64, f64);
type ComplexTrace = Vec<ComplexTraceSample>;
type SensitivityTracePair = (ComplexTrace, ComplexTrace);

fn peak(samples: &[ComplexTraceSample]) -> ResponsePeak {
    let (time, real, imaginary) = samples
        .iter()
        .copied()
        .max_by(|left, right| left.1.hypot(left.2).total_cmp(&right.1.hypot(right.2)))
        .unwrap_or((0.0, 0.0, 0.0));
    ResponsePeak {
        time,
        magnitude: real.hypot(imaginary),
        phase: imaginary.atan2(real),
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

fn sensitivity_at_time(samples: &[ComplexTraceSample], target_time: f64) -> ComplexTraceSample {
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
    let second_kick: Vec<f64> = (0..config.nx)
        .map(|ix| {
            SECOND_PULSE_AMPLITUDE
                * (TAU * SECOND_MODE as f64 * config.x(ix) / config.domain_length).sin()
        })
        .collect();
    checkpoint_solver.apply_velocity_impulse(&second_kick)?;
    let checkpoint = checkpoint_solver.distribution().to_vec();

    let checkpoint_diagnostics = checkpoint_solver.diagnostics()?;
    println!(
        "spatial_mode,velocity_mode,epsilon,minimum_plus,minimum_minus,max_moment_residual,ballistic_rephase_time,target_dE_rephase_magnitude,target_dE_rephase_phase,target_peak_time,target_peak_sensitivity,target_peak_phase,intervention_mode_peak_sensitivity,intervention_to_echo_sensitivity_ratio,epsilon_half_peak_sensitivity,epsilon_half_peak_relative_delta"
    );

    for velocity_mode in VELOCITY_MODES {
        let (direction, epsilon, moment_residual) =
            moment_neutral_direction(&checkpoint, config, FIRST_MODE, velocity_mode)?;
        let plus_minimum = checkpoint
            .iter()
            .zip(&direction)
            .map(|(value, delta)| value + epsilon * delta)
            .fold(f64::INFINITY, f64::min);
        let minus_minimum = checkpoint
            .iter()
            .zip(&direction)
            .map(|(value, delta)| value - epsilon * delta)
            .fold(f64::INFINITY, f64::min);

        let (target_trace, off_target_trace) =
            central_difference_trace(config, &checkpoint, &direction, epsilon, future_steps, 2)?;
        let target_peak = peak(&target_trace);
        let off_target_peak = peak(&off_target_trace);
        let rephase = sensitivity_at_time(&target_trace, 2.0);
        let epsilon_half = 0.5 * epsilon;
        let (half_trace, _) = central_difference_trace(
            config,
            &checkpoint,
            &direction,
            epsilon_half,
            future_steps,
            2,
        )?;
        let half_peak = peak(&half_trace);
        let half_relative_delta = (half_peak.magnitude - target_peak.magnitude).abs()
            / target_peak.magnitude.max(f64::MIN_POSITIVE);
        let off_target_ratio =
            off_target_peak.magnitude / target_peak.magnitude.max(f64::MIN_POSITIVE);

        println!(
            "{},{},{:.12e},{:.12e},{:.12e},{:.12e},4.0,{:.12e},{:.12e},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            FIRST_MODE,
            velocity_mode,
            epsilon,
            plus_minimum,
            minus_minimum,
            moment_residual,
            rephase.1.hypot(rephase.2),
            rephase.2.atan2(rephase.1),
            target_peak.time,
            target_peak.magnitude,
            target_peak.phase,
            off_target_peak.magnitude,
            off_target_ratio,
            half_peak.magnitude,
            half_relative_delta,
        );
    }

    let _ = checkpoint_diagnostics;
    Ok(())
}
