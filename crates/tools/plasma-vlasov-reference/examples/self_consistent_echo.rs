use std::error::Error;
use std::f64::consts::TAU;

use plasma_vlasov_reference::{
    EchoMeasurementConfig, EchoModeTrace, EchoPulsePair, EchoTraceSample, VlasovPoisson1d1v,
    VlasovPoissonConfig, echo_analysis::measure_echo_trace, maxwellian_perturbation,
};

const FIRST_MODE: usize = 1;
const SECOND_MODE: usize = 2;
const FIRST_TIME: f64 = 0.0;
const SECOND_TIME: f64 = 2.0;
const FINAL_TIME: f64 = 5.0;
const INITIAL_AMPLITUDE: f64 = 1.0e-3;
const THERMAL_SPEED: f64 = 1.0;
const IMPULSE_AMPLITUDE: f64 = 5.0e-2;
const ECHO_SIGNAL_FLOOR: f64 = 1.0e-9;

fn run_resolution(
    label: &str,
    config: VlasovPoissonConfig,
    dt: f64,
    sample_every: usize,
) -> Result<(), Box<dyn Error>> {
    let pulse_pair = EchoPulsePair {
        first_mode: FIRST_MODE as i32,
        second_mode: SECOND_MODE as i32,
        first_time: FIRST_TIME,
        second_time: SECOND_TIME,
    };
    let prediction = plasma_vlasov_reference::echo_analysis::predict_two_pulse_echo(pulse_pair)?;

    let initial = maxwellian_perturbation(
        config,
        INITIAL_AMPLITUDE,
        FIRST_MODE,
        THERMAL_SPEED,
    )?;
    let mut pulsed = VlasovPoisson1d1v::new(config, initial.clone())?;
    let mut no_second_pulse = VlasovPoisson1d1v::new(config, initial)?;

    let second_kick: Vec<f64> = (0..config.nx)
        .map(|ix| {
            let phase =
                TAU * SECOND_MODE as f64 * config.x(ix) / config.domain_length;
            IMPULSE_AMPLITUDE * phase.sin()
        })
        .collect();

    let pulse_step = (SECOND_TIME / dt).round() as usize;
    let total_steps = (FINAL_TIME / dt).round() as usize;
    if pulse_step == 0
        || total_steps <= pulse_step
        || (pulse_step as f64 * dt - SECOND_TIME).abs() > 1.0e-12
        || (total_steps as f64 * dt - FINAL_TIME).abs() > 1.0e-12
        || sample_every == 0
    {
        return Err("pulse/final times must be represented exactly by the step grid".into());
    }

    let echo_window_start = prediction.echo_time - 1.0;
    let echo_window_end = prediction.echo_time + 1.0;
    let mut echo_trace = Vec::new();

    for step in 0..=total_steps {
        if step == pulse_step {
            // The pulse is an instantaneous, externally prescribed intervention.
            // Every other interval is advanced with the self-consistent Vlasov-
            // Poisson Strang-split step operator.
            pulsed.apply_velocity_impulse(&second_kick)?;
        }

        if step % sample_every == 0 || step == pulse_step || step == total_steps {
            let time = pulsed.time();
            let pulse_echo = pulsed.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let control_echo =
                no_second_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let delta_real = pulse_echo.real - control_echo.real;
            let delta_imaginary = pulse_echo.imaginary - control_echo.imaginary;
            let pulse_second = pulsed.electric_mode_coefficient(SECOND_MODE)?;
            let control_second =
                no_second_pulse.electric_mode_coefficient(SECOND_MODE)?;
            let pulse_diagnostics = pulsed.diagnostics()?;
            let control_diagnostics = no_second_pulse.diagnostics()?;
            let pulse_tails = pulsed.spectral_tail_diagnostics(0.75)?;

            println!(
                "{label},{},{},{dt:.8},{step},{time:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                config.nx,
                config.nv,
                pulse_echo.real,
                pulse_echo.imaginary,
                pulse_echo.amplitude(),
                control_echo.real,
                control_echo.imaginary,
                control_echo.amplitude(),
                delta_real,
                delta_imaginary,
                delta_real.hypot(delta_imaginary),
                pulse_second.real,
                pulse_second.imaginary,
                pulse_second.amplitude(),
                control_second.real,
                control_second.imaginary,
                control_second.amplitude(),
                pulse_diagnostics.total_mass,
                pulse_diagnostics.momentum,
                pulse_diagnostics.kinetic_energy,
                pulse_diagnostics.field_energy,
                pulse_diagnostics.total_energy,
                pulse_diagnostics.max_poisson_residual,
                pulse_diagnostics.minimum_distribution,
                pulse_diagnostics.velocity_boundary_ratio,
                pulse_tails.x.high_frequency_fraction,
                pulse_tails.velocity.high_frequency_fraction,
                pulse_tails.x.nyquist_energy_fraction,
                pulse_tails.velocity.nyquist_energy_fraction,
                control_diagnostics.total_mass,
                control_diagnostics.momentum,
                control_diagnostics.total_energy,
                control_diagnostics.field_energy,
                control_diagnostics.max_poisson_residual,
                control_diagnostics.minimum_distribution,
                control_diagnostics.velocity_boundary_ratio,
                pulse_diagnostics.time,
            );

            if (echo_window_start..=echo_window_end).contains(&time) {
                echo_trace.push(EchoTraceSample::new(time, delta_real, delta_imaginary));
            }
        }

        if step < total_steps {
            pulsed.step(dt)?;
            no_second_pulse.step(dt)?;
        }
    }

    let measurement = measure_echo_trace(
        prediction,
        EchoModeTrace {
            spatial_mode: prediction.echo_mode,
            samples: &echo_trace,
        },
        EchoMeasurementConfig {
            minimum_peak_amplitude: ECHO_SIGNAL_FLOOR,
        },
    )?;

    println!(
        "# summary,{label},{},{},{dt:.8},echo_mode={},predicted_echo_time={:.8},sampled_peak_time={:.8},peak_amplitude={:.12e},peak_phase={:.12e},detected={},fwhm={}",
        config.nx,
        config.nv,
        prediction.echo_mode,
        prediction.echo_time,
        measurement.sampled_peak_time,
        measurement.peak_amplitude,
        measurement.peak_phase_radians,
        measurement.detected,
        measurement
            .full_width_half_maximum
            .map(|width| format!("{width:.8}"))
            .unwrap_or_else(|| "unbracketed".to_string()),
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "run,nx,nv,dt,step,time,pulse_echo_real,pulse_echo_imag,pulse_echo_amplitude,control_echo_real,control_echo_imag,control_echo_amplitude,delta_echo_real,delta_echo_imag,delta_echo_amplitude,pulse_second_real,pulse_second_imag,pulse_second_amplitude,control_second_real,control_second_imag,control_second_amplitude,pulse_mass,pulse_momentum,pulse_kinetic_energy,pulse_field_energy,pulse_total_energy,pulse_poisson_residual,pulse_minimum_f,pulse_velocity_boundary_ratio,pulse_x_high_tail_fraction,pulse_v_high_tail_fraction,pulse_x_nyquist_fraction,pulse_v_nyquist_fraction,control_mass,control_momentum,control_total_energy,control_field_energy,control_poisson_residual,control_minimum_f,control_velocity_boundary_ratio,diagnostic_time"
    );

    let common = |nx, nv| VlasovPoissonConfig {
        nx,
        nv,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -6.0,
        velocity_max: 6.0,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    };

    // This is a two-level convergence probe, not by itself a convergence proof.
    run_resolution("coarse", common(64, 256), 0.01, 4)?;
    run_resolution("fine", common(128, 512), 0.005, 8)?;
    Ok(())
}
