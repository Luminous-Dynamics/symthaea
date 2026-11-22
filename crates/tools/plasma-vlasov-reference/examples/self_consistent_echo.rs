use std::error::Error;
use std::f64::consts::TAU;

use plasma_vlasov_reference::echo_analysis::{
    EchoMeasurementConfig, EchoModeTrace, EchoPulsePair, EchoTraceSample, measure_echo_trace,
};
use plasma_vlasov_reference::phase_space_cascade::JointPhaseSpaceSpectrum;
use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

const FIRST_MODE: usize = 1;
const SECOND_MODE: usize = 2;
const FIRST_TIME: f64 = 0.0;
const SECOND_TIME: f64 = 2.0;
const FINAL_TIME: f64 = 12.0;
const INITIAL_AMPLITUDE: f64 = 1.0e-3;
const THERMAL_SPEED: f64 = 1.0;
const IMPULSE_AMPLITUDE: f64 = 5.0e-2;
const ECHO_SIGNAL_FLOOR: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy)]
struct EchoRidgeMetrics {
    total: f64,
    positive_s_fraction: f64,
    negative_s_fraction: f64,
    zero_s_fraction: f64,
    nyquist_s_fraction: f64,
    signed_s_centroid: f64,
    dominant_signed_s: isize,
}

fn echo_mode_ridge(spectrum: &JointPhaseSpaceSpectrum, signed_x_mode: isize) -> EchoRidgeMetrics {
    let x_mode_index = if signed_x_mode >= 0 {
        signed_x_mode as usize
    } else {
        (spectrum.nx as isize + signed_x_mode) as usize
    };
    if x_mode_index >= spectrum.nx {
        return EchoRidgeMetrics {
            total: 0.0,
            positive_s_fraction: 0.0,
            negative_s_fraction: 0.0,
            zero_s_fraction: 0.0,
            nyquist_s_fraction: 0.0,
            signed_s_centroid: 0.0,
            dominant_signed_s: isize::MIN,
        };
    }

    let mut total = 0.0;
    let mut positive_s = 0.0;
    let mut negative_s = 0.0;
    let mut zero_s = 0.0;
    let mut nyquist_s = 0.0;
    let mut signed_s_moment = 0.0;
    let mut non_nyquist_total = 0.0;
    let mut dominant_energy = -1.0_f64;
    let mut dominant_signed_s = isize::MIN;

    for velocity_index in 0..spectrum.nv {
        let energy = spectrum
            .mode_energy(x_mode_index, velocity_index)
            .unwrap_or(0.0);
        total += energy;
        if velocity_index == spectrum.nv / 2 {
            nyquist_s += energy;
            if energy > dominant_energy {
                dominant_energy = energy;
                dominant_signed_s = isize::MIN;
            }
            continue;
        }

        let signed_s = spectrum
            .signed_velocity_mode(velocity_index)
            .unwrap_or_default();
        non_nyquist_total += energy;
        signed_s_moment += signed_s as f64 * energy;
        if energy > dominant_energy {
            dominant_energy = energy;
            dominant_signed_s = signed_s;
        }
        if signed_s > 0 {
            positive_s += energy;
        } else if signed_s < 0 {
            negative_s += energy;
        } else {
            zero_s += energy;
        }
    }

    let denominator = total.max(f64::MIN_POSITIVE);
    EchoRidgeMetrics {
        total,
        positive_s_fraction: positive_s / denominator,
        negative_s_fraction: negative_s / denominator,
        zero_s_fraction: zero_s / denominator,
        nyquist_s_fraction: nyquist_s / denominator,
        signed_s_centroid: if non_nyquist_total > 0.0 {
            signed_s_moment / non_nyquist_total
        } else {
            0.0
        },
        dominant_signed_s,
    }
}

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

    let initial = maxwellian_perturbation(config, INITIAL_AMPLITUDE, FIRST_MODE, THERMAL_SPEED)?;
    let mut pulsed = VlasovPoisson1d1v::new(config, initial.clone())?;
    let mut no_second_pulse = VlasovPoisson1d1v::new(config, initial)?;
    let initial_casimir = pulsed.quadratic_casimir_budget()?.total_casimir;

    let second_kick: Vec<f64> = (0..config.nx)
        .map(|ix| {
            let phase = TAU * SECOND_MODE as f64 * config.x(ix) / config.domain_length;
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

    let echo_window_start = prediction.echo_time - 1.5;
    let echo_window_end = prediction.echo_time + 7.5;
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
            let control_second = no_second_pulse.electric_mode_coefficient(SECOND_MODE)?;
            let pulse_diagnostics = pulsed.diagnostics()?;
            let control_diagnostics = no_second_pulse.diagnostics()?;
            let pulse_tails = pulsed.spectral_tail_diagnostics(0.75)?;

            println!(
                "{label},{},{},{dt:.8},{step},{time:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
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

        let cascade_every = sample_every.saturating_mul(10).max(1);
        if step % cascade_every == 0 || step == pulse_step || step == total_steps {
            let pulse_budget = pulsed.quadratic_casimir_budget()?;
            let control_budget = no_second_pulse.quadratic_casimir_budget()?;
            let pulse_spectrum = pulsed.joint_phase_space_spectrum()?;
            let control_spectrum = no_second_pulse.joint_phase_space_spectrum()?;
            let pulse_scale = pulse_spectrum.spectral_f_strophy.max(f64::MIN_POSITIVE);
            let control_scale = control_spectrum.spectral_f_strophy.max(f64::MIN_POSITIVE);
            let pulse_ridge = echo_mode_ridge(&pulse_spectrum, prediction.echo_mode as isize);
            let control_ridge = echo_mode_ridge(&control_spectrum, prediction.echo_mode as isize);
            let response_distribution: Vec<f64> = pulsed
                .distribution()
                .iter()
                .zip(no_second_pulse.distribution())
                .map(|(pulse, control)| pulse - control)
                .collect();
            let response_state = VlasovPoisson1d1v::new(config, response_distribution)?;
            let response_spectrum = response_state.joint_phase_space_spectrum()?;
            let response_ridge =
                echo_mode_ridge(&response_spectrum, prediction.echo_mode as isize);

            println!(
                "# ridge,{label},{step},{:.8},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{}",
                pulsed.time(),
                prediction.echo_mode,
                pulse_ridge.total,
                pulse_ridge.positive_s_fraction,
                pulse_ridge.negative_s_fraction,
                pulse_ridge.zero_s_fraction,
                pulse_ridge.nyquist_s_fraction,
                pulse_ridge.signed_s_centroid,
                pulse_ridge.dominant_signed_s,
                control_ridge.total,
                control_ridge.positive_s_fraction,
                control_ridge.negative_s_fraction,
                control_ridge.zero_s_fraction,
                control_ridge.nyquist_s_fraction,
                control_ridge.signed_s_centroid,
                control_ridge.dominant_signed_s,
                response_ridge.total,
                response_ridge.positive_s_fraction,
                response_ridge.negative_s_fraction,
                response_ridge.zero_s_fraction,
                response_ridge.nyquist_s_fraction,
                response_ridge.signed_s_centroid,
                response_ridge.dominant_signed_s,
            );

            println!(
                "# cascade,{label},{step},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                pulsed.time(),
                pulse_budget.total_casimir,
                control_budget.total_casimir,
                (pulse_budget.total_casimir - initial_casimir).abs()
                    / initial_casimir.max(f64::MIN_POSITIVE),
                (control_budget.total_casimir - initial_casimir).abs()
                    / initial_casimir.max(f64::MIN_POSITIVE),
                pulse_budget.spatial_mean_casimir,
                pulse_budget.fluctuation_f_strophy,
                pulse_budget.fluctuation_f_strophy
                    / pulse_budget.total_casimir.max(f64::MIN_POSITIVE),
                pulse_spectrum.phase_mixing_f_strophy / pulse_scale,
                pulse_spectrum.phase_unmixing_f_strophy / pulse_scale,
                pulse_spectrum.zero_axis_f_strophy / pulse_scale,
                pulse_spectrum.nyquist_f_strophy / pulse_scale,
                pulse_spectrum.parseval_relative_residual,
                control_budget.spatial_mean_casimir,
                control_budget.fluctuation_f_strophy,
                control_spectrum.phase_mixing_f_strophy / control_scale,
                control_spectrum.phase_unmixing_f_strophy / control_scale,
                control_spectrum.zero_axis_f_strophy / control_scale,
                control_spectrum.nyquist_f_strophy / control_scale,
                control_spectrum.parseval_relative_residual,
            );
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

    println!(
        "# cascade_columns,run,step,time,pulse_C2,control_C2,pulse_C2_relative_drift,control_C2_relative_drift,pulse_mean_C2,pulse_f_strophy,pulse_f_strophy_fraction,pulse_mixing_fraction,pulse_unmixing_fraction,pulse_zero_axis_fraction,pulse_nyquist_fraction,pulse_parseval_relative,control_mean_C2,control_f_strophy,control_mixing_fraction,control_unmixing_fraction,control_zero_axis_fraction,control_nyquist_fraction,control_parseval_relative"
    );

    println!(
        "# ridge_columns,run,step,time,mode,pulse_k_mode_f_strophy,pulse_positive_s_fraction,pulse_negative_s_fraction,pulse_zero_s_fraction,pulse_nyquist_s_fraction,pulse_signed_s_centroid,pulse_dominant_signed_s,control_k_mode_f_strophy,control_positive_s_fraction,control_negative_s_fraction,control_zero_s_fraction,control_nyquist_s_fraction,control_signed_s_centroid,control_dominant_signed_s,response_k_mode_f_strophy,response_positive_s_fraction,response_negative_s_fraction,response_zero_s_fraction,response_nyquist_s_fraction,response_signed_s_centroid,response_dominant_signed_s"
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

    // One-factor-at-a-time refinements isolate spatial grid, velocity grid and
    // timestep effects. The combined fine case checks their interaction.
    // All configurations emit at the same 0.02 physical-time cadence.
    run_resolution("base", common(64, 256), 0.01, 2)?;
    run_resolution("x_refined", common(128, 256), 0.01, 2)?;
    run_resolution("v_refined", common(64, 512), 0.01, 2)?;
    run_resolution("dt_refined", common(64, 256), 0.005, 4)?;
    run_resolution("joint_refined", common(128, 512), 0.005, 4)?;
    Ok(())
}
