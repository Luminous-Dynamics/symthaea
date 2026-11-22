use std::error::Error;
use std::f64::consts::TAU;
use std::time::Instant;

use plasma_vlasov_reference::echo_analysis::{
    EchoMeasurementConfig, EchoModeTrace, EchoPulsePair, EchoTraceSample, measure_echo_trace,
};
use plasma_vlasov_reference::phase_space_cascade::JointPhaseSpaceSpectrum;
use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

const FIRST_MODE: usize = 1;
const SECOND_MODE: usize = 2;
const WRONG_SECOND_MODE: usize = 3;
const FIRST_TIME: f64 = 0.0;
const SECOND_TIME: f64 = 2.0;
const FINAL_TIME: f64 = 12.0;
const INITIAL_AMPLITUDE: f64 = 1.0e-3;
const THERMAL_SPEED: f64 = 1.0;
const ECHO_SIGNAL_FLOOR: f64 = 1.0e-9;
const FLUX_X_MODE_CUTOFF: usize = 3;
const FLUX_VELOCITY_MODE_CUTOFF: usize = 2;

#[derive(Debug, Clone, Copy)]
struct EchoRidgeMetrics {
    total: f64,
    positive_s_fraction: f64,
    negative_s_fraction: f64,
    zero_s_fraction: f64,
    nyquist_s_fraction: f64,
    signed_velocity_mode_index_centroid: f64,
    dominant_signed_velocity_mode_index: isize,
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
            signed_velocity_mode_index_centroid: 0.0,
            dominant_signed_velocity_mode_index: isize::MIN,
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
    let mut dominant_signed_velocity_mode_index = isize::MIN;

    for velocity_index in 0..spectrum.nv {
        let energy = spectrum
            .mode_energy(x_mode_index, velocity_index)
            .unwrap_or(0.0);
        total += energy;
        if velocity_index == spectrum.nv / 2 {
            nyquist_s += energy;
            if energy > dominant_energy {
                dominant_energy = energy;
                dominant_signed_velocity_mode_index = isize::MIN;
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
            dominant_signed_velocity_mode_index = signed_s;
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
        signed_velocity_mode_index_centroid: if non_nyquist_total > 0.0 {
            signed_s_moment / non_nyquist_total
        } else {
            0.0
        },
        dominant_signed_velocity_mode_index,
    }
}

fn low_x_f_strophy(spectrum: &JointPhaseSpaceSpectrum, cutoff: usize) -> f64 {
    spectrum
        .by_x_mode
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            spectrum
                .signed_x_mode(*index)
                .unwrap_or_default()
                .unsigned_abs()
                <= cutoff
        })
        .map(|(_, value)| *value)
        .sum()
}

fn low_velocity_f_strophy(spectrum: &JointPhaseSpaceSpectrum, cutoff: usize) -> f64 {
    spectrum
        .by_velocity_mode
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            spectrum
                .signed_velocity_mode(*index)
                .unwrap_or_default()
                .unsigned_abs()
                <= cutoff
        })
        .map(|(_, value)| *value)
        .sum()
}

fn integrate_quadratic_flux_triplet(
    start_time: f64,
    middle_time: f64,
    end_time: f64,
    start_rate: f64,
    middle_rate: f64,
    end_rate: f64,
) -> f64 {
    let first_interval = middle_time - start_time;
    let second_interval = end_time - middle_time;
    let span = first_interval + second_interval;
    let start_weight = span * (2.0 * first_interval - second_interval) / (6.0 * first_interval);
    let middle_weight = span.powi(3) / (6.0 * first_interval * second_interval);
    let end_weight = span * (2.0 * second_interval - first_interval) / (6.0 * second_interval);
    start_weight * start_rate + middle_weight * middle_rate + end_weight * end_rate
}

fn run_resolution(
    label: &str,
    config: VlasovPoissonConfig,
    dt: f64,
    sample_every: usize,
    impulse_amplitude: f64,
) -> Result<(), Box<dyn Error>> {
    let wall_clock_start = Instant::now();
    let pulse_pair = EchoPulsePair {
        first_mode: FIRST_MODE as i32,
        second_mode: SECOND_MODE as i32,
        first_time: FIRST_TIME,
        second_time: SECOND_TIME,
    };
    let prediction = plasma_vlasov_reference::echo_analysis::predict_two_pulse_echo(pulse_pair)?;
    let wrong_pulse_pair = EchoPulsePair {
        first_mode: FIRST_MODE as i32,
        second_mode: WRONG_SECOND_MODE as i32,
        first_time: FIRST_TIME,
        second_time: SECOND_TIME,
    };
    let wrong_prediction =
        plasma_vlasov_reference::echo_analysis::predict_two_pulse_echo(wrong_pulse_pair)?;
    let velocity_spacing = config.dv();
    let fundamental_wave_number = FIRST_MODE as f64 * TAU / config.domain_length;
    let velocity_nyquist_time = PI / (fundamental_wave_number * velocity_spacing);
    let velocity_recurrence_time = TAU / (fundamental_wave_number * velocity_spacing);

    let initial = maxwellian_perturbation(config, INITIAL_AMPLITUDE, FIRST_MODE, THERMAL_SPEED)?;
    let mut pulsed = VlasovPoisson1d1v::new(config, initial.clone())?;
    let mut no_second_pulse = VlasovPoisson1d1v::new(config, initial.clone())?;
    let mut wrong_mode = VlasovPoisson1d1v::new(config, initial)?;
    let initial_casimir = pulsed.quadratic_casimir_budget()?.total_casimir;
    let initial_spectrum = pulsed.joint_phase_space_spectrum()?;
    let mut pulse_flux_baseline_x = low_x_f_strophy(&initial_spectrum, FLUX_X_MODE_CUTOFF);
    let mut pulse_flux_baseline_v =
        low_velocity_f_strophy(&initial_spectrum, FLUX_VELOCITY_MODE_CUTOFF);
    let mut control_flux_baseline_x = pulse_flux_baseline_x;
    let mut control_flux_baseline_v = pulse_flux_baseline_v;
    let mut pulse_cumulative_flux_x = 0.0;
    let mut pulse_cumulative_flux_v = 0.0;
    let mut control_cumulative_flux_x = 0.0;
    let mut control_cumulative_flux_v = 0.0;
    // Composite three-point quadratic integration avoids the 1-2% bias
    // observed with trapezoidal integration at 0.2-time-unit flux cadence.
    // Rebase at the t2+ impulse boundary and integrate non-overlapping pairs.
    type FluxSample = (f64, f64, f64, f64, f64);
    let mut flux_pair_start: Option<FluxSample> = None;
    let mut flux_pair_middle: Option<FluxSample> = None;

    let second_kick: Vec<f64> = (0..config.nx)
        .map(|ix| {
            let phase = TAU * SECOND_MODE as f64 * config.x(ix) / config.domain_length;
            impulse_amplitude * phase.sin()
        })
        .collect();
    let wrong_second_kick: Vec<f64> = (0..config.nx)
        .map(|ix| {
            let phase = TAU * WRONG_SECOND_MODE as f64 * config.x(ix) / config.domain_length;
            impulse_amplitude * phase.sin()
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
    let wrong_echo_window_start = wrong_prediction.echo_time - 1.5;
    let wrong_echo_window_end = wrong_prediction.echo_time + 7.5;
    let mut echo_trace = Vec::new();
    let mut wrong_echo_trace = Vec::new();
    let mut wrong_mode_off_target_peak = 0.0_f64;
    let mut wrong_mode_off_target_peak_time = FIRST_TIME;

    for step in 0..=total_steps {
        if step == pulse_step {
            // The pulse is an instantaneous, externally prescribed intervention.
            // Every other interval is advanced with the self-consistent Vlasov-
            // Poisson Strang-split step operator.
            pulsed.apply_velocity_impulse(&second_kick)?;
            wrong_mode.apply_velocity_impulse(&wrong_second_kick)?;
        }

        if step % sample_every == 0 || step == pulse_step || step == total_steps {
            let time = pulsed.time();
            let pulse_echo = pulsed.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let control_echo =
                no_second_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let delta_real = pulse_echo.real - control_echo.real;
            let delta_imaginary = pulse_echo.imaginary - control_echo.imaginary;
            let wrong_echo =
                wrong_mode.electric_mode_coefficient(wrong_prediction.echo_mode as usize)?;
            let wrong_control_echo =
                no_second_pulse.electric_mode_coefficient(wrong_prediction.echo_mode as usize)?;
            let wrong_delta_real = wrong_echo.real - wrong_control_echo.real;
            let wrong_delta_imaginary = wrong_echo.imaginary - wrong_control_echo.imaginary;
            let wrong_off_target =
                wrong_mode.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let wrong_off_target_control =
                no_second_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let wrong_off_target_amplitude = (wrong_off_target.real
                - wrong_off_target_control.real)
                .hypot(wrong_off_target.imaginary - wrong_off_target_control.imaginary);
            if wrong_off_target_amplitude > wrong_mode_off_target_peak {
                wrong_mode_off_target_peak = wrong_off_target_amplitude;
                wrong_mode_off_target_peak_time = time;
            }
            let pulse_second = pulsed.electric_mode_coefficient(SECOND_MODE)?;
            let control_second = no_second_pulse.electric_mode_coefficient(SECOND_MODE)?;
            let pulse_diagnostics = pulsed.diagnostics()?;
            let control_diagnostics = no_second_pulse.diagnostics()?;
            let pulse_tails = pulsed.spectral_tail_diagnostics(0.75)?;

            println!(
                "{label},{},{},{dt:.8},{impulse_amplitude:.8},{step},{time:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
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
            if (wrong_echo_window_start..=wrong_echo_window_end).contains(&time) {
                wrong_echo_trace.push(EchoTraceSample::new(
                    time,
                    wrong_delta_real,
                    wrong_delta_imaginary,
                ));
            }
            println!(
                "# negative_control,{label},{step},{time:.8},{},{},{impulse_amplitude:.8},{:.12e},{:.8},{:.12e},{:.8},{:.12e},{:.8},{:.12e}",
                WRONG_SECOND_MODE,
                wrong_prediction.echo_mode,
                wrong_delta_real.hypot(wrong_delta_imaginary),
                wrong_delta_imaginary.atan2(wrong_delta_real),
                wrong_echo.amplitude(),
                wrong_prediction.echo_time,
                wrong_mode_off_target_peak,
                wrong_mode_off_target_peak_time,
                wrong_off_target_amplitude,
            );
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
            let response_ridge = echo_mode_ridge(&response_spectrum, prediction.echo_mode as isize);
            let pulse_flux =
                pulsed.phase_space_transfer_rates(FLUX_X_MODE_CUTOFF, FLUX_VELOCITY_MODE_CUTOFF)?;
            let control_flux = no_second_pulse
                .phase_space_transfer_rates(FLUX_X_MODE_CUTOFF, FLUX_VELOCITY_MODE_CUTOFF)?;
            let pulse_low_x = low_x_f_strophy(&pulse_spectrum, FLUX_X_MODE_CUTOFF);
            let pulse_low_v = low_velocity_f_strophy(&pulse_spectrum, FLUX_VELOCITY_MODE_CUTOFF);
            let control_low_x = low_x_f_strophy(&control_spectrum, FLUX_X_MODE_CUTOFF);
            let control_low_v =
                low_velocity_f_strophy(&control_spectrum, FLUX_VELOCITY_MODE_CUTOFF);
            let flux_time = pulsed.time();

            // The instantaneous impulse at t2 is a discontinuous intervention,
            // not a continuous Vlasov flux. Rebaseline both trajectories at t2+
            // so the integrated flux closes the smooth post-intervention leg.
            let current_flux_sample = (
                flux_time,
                pulse_flux.flux_out_of_low_x_modes,
                pulse_flux.flux_out_of_low_velocity_modes,
                control_flux.flux_out_of_low_x_modes,
                control_flux.flux_out_of_low_velocity_modes,
            );

            if step == pulse_step {
                pulse_flux_baseline_x = pulse_low_x;
                pulse_flux_baseline_v = pulse_low_v;
                control_flux_baseline_x = control_low_x;
                control_flux_baseline_v = control_low_v;
                pulse_cumulative_flux_x = 0.0;
                pulse_cumulative_flux_v = 0.0;
                control_cumulative_flux_x = 0.0;
                control_cumulative_flux_v = 0.0;
                flux_pair_start = Some(current_flux_sample);
                flux_pair_middle = None;
            } else if let (Some(start), Some(middle)) = (flux_pair_start, flux_pair_middle) {
                pulse_cumulative_flux_x += integrate_quadratic_flux_triplet(
                    start.0,
                    middle.0,
                    flux_time,
                    start.1,
                    middle.1,
                    pulse_flux.flux_out_of_low_x_modes,
                );
                pulse_cumulative_flux_v += integrate_quadratic_flux_triplet(
                    start.0,
                    middle.0,
                    flux_time,
                    start.2,
                    middle.2,
                    pulse_flux.flux_out_of_low_velocity_modes,
                );
                control_cumulative_flux_x += integrate_quadratic_flux_triplet(
                    start.0,
                    middle.0,
                    flux_time,
                    start.3,
                    middle.3,
                    control_flux.flux_out_of_low_x_modes,
                );
                control_cumulative_flux_v += integrate_quadratic_flux_triplet(
                    start.0,
                    middle.0,
                    flux_time,
                    start.4,
                    middle.4,
                    control_flux.flux_out_of_low_velocity_modes,
                );
                flux_pair_start = Some(current_flux_sample);
                flux_pair_middle = None;
            } else if flux_pair_start.is_some() {
                flux_pair_middle = Some(current_flux_sample);
            } else {
                flux_pair_start = Some(current_flux_sample);
            }
            let pulse_direct_loss_x = pulse_flux_baseline_x - pulse_low_x;
            let pulse_direct_loss_v = pulse_flux_baseline_v - pulse_low_v;
            let control_direct_loss_x = control_flux_baseline_x - control_low_x;
            let control_direct_loss_v = control_flux_baseline_v - control_low_v;

            println!(
                "# flux,{label},{step},{:.8},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                flux_time,
                FLUX_X_MODE_CUTOFF,
                FLUX_VELOCITY_MODE_CUTOFF,
                pulse_flux.flux_out_of_low_x_modes,
                pulse_flux.flux_out_of_low_velocity_modes,
                pulse_cumulative_flux_x,
                pulse_cumulative_flux_v,
                pulse_direct_loss_x,
                pulse_direct_loss_v,
                pulse_cumulative_flux_x - pulse_direct_loss_x,
                pulse_cumulative_flux_v - pulse_direct_loss_v,
                control_flux.flux_out_of_low_x_modes,
                control_flux.flux_out_of_low_velocity_modes,
                control_cumulative_flux_x,
                control_cumulative_flux_v,
                control_direct_loss_x,
                control_direct_loss_v,
                control_cumulative_flux_x - control_direct_loss_x,
                control_cumulative_flux_v - control_direct_loss_v,
            );

            println!(
                "# ridge,{label},{step},{:.8},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{}",
                pulsed.time(),
                prediction.echo_mode,
                pulse_ridge.total,
                pulse_ridge.positive_s_fraction,
                pulse_ridge.negative_s_fraction,
                pulse_ridge.zero_s_fraction,
                pulse_ridge.nyquist_s_fraction,
                pulse_ridge.signed_velocity_mode_index_centroid,
                pulse_ridge.dominant_signed_velocity_mode_index,
                control_ridge.total,
                control_ridge.positive_s_fraction,
                control_ridge.negative_s_fraction,
                control_ridge.zero_s_fraction,
                control_ridge.nyquist_s_fraction,
                control_ridge.signed_velocity_mode_index_centroid,
                control_ridge.dominant_signed_velocity_mode_index,
                response_ridge.total,
                response_ridge.positive_s_fraction,
                response_ridge.negative_s_fraction,
                response_ridge.zero_s_fraction,
                response_ridge.nyquist_s_fraction,
                response_ridge.signed_velocity_mode_index_centroid,
                response_ridge.dominant_signed_velocity_mode_index,
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
            wrong_mode.step(dt)?;
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

    let wrong_measurement = measure_echo_trace(
        wrong_prediction,
        EchoModeTrace {
            spatial_mode: wrong_prediction.echo_mode,
            samples: &wrong_echo_trace,
        },
        EchoMeasurementConfig {
            minimum_peak_amplitude: ECHO_SIGNAL_FLOOR,
        },
    )?;

    println!(
        "# summary,{label},{},{},{dt:.8},{impulse_amplitude:.8},dv={:.8},velocity_nyquist_time={:.8},velocity_recurrence_time={:.8},echo_mode={},predicted_echo_time={:.8},sampled_peak_time={:.8},peak_amplitude={:.12e},peak_phase={:.12e},detected={},fwhm={},wall_seconds={:.6}",
        config.nx,
        config.nv,
        velocity_spacing,
        velocity_nyquist_time,
        velocity_recurrence_time,
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
        wall_clock_start.elapsed().as_secs_f64(),
    );
    println!(
        "# negative_summary,{label},{impulse_amplitude:.8},wrong_pulse_mode={},predicted_echo_mode={},predicted_rephasing_time={:.8},sampled_wrong_mode_peak_time={:.8},wrong_mode_peak_amplitude={:.12e},wrong_mode_peak_phase={:.12e},wrong_mode_detected={},off_target_original_mode_peak={:.12e},off_target_peak_time={:.8}",
        WRONG_SECOND_MODE,
        wrong_prediction.echo_mode,
        wrong_prediction.echo_time,
        wrong_measurement.sampled_peak_time,
        wrong_measurement.peak_amplitude,
        wrong_measurement.peak_phase_radians,
        wrong_measurement.detected,
        wrong_mode_off_target_peak,
        wrong_mode_off_target_peak_time,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::integrate_quadratic_flux_triplet;

    fn polynomial(time: f64) -> f64 {
        2.0 * time * time - 3.0 * time + 4.0
    }

    fn antiderivative(time: f64) -> f64 {
        (2.0 / 3.0) * time.powi(3) - 1.5 * time.powi(2) + 4.0 * time
    }

    #[test]
    fn quadratic_flux_integrator_is_exact_for_uniform_samples() {
        let observed = integrate_quadratic_flux_triplet(
            0.0,
            1.0,
            2.0,
            polynomial(0.0),
            polynomial(1.0),
            polynomial(2.0),
        );
        let expected = antiderivative(2.0) - antiderivative(0.0);
        assert!((observed - expected).abs() < 1.0e-13);
    }

    #[test]
    fn quadratic_flux_integrator_is_exact_for_nonuniform_samples() {
        let start = -1.0;
        let middle = 0.5;
        let end = 3.0;
        let observed = integrate_quadratic_flux_triplet(
            start,
            middle,
            end,
            polynomial(start),
            polynomial(middle),
            polynomial(end),
        );
        let expected = antiderivative(end) - antiderivative(start);
        assert!((observed - expected).abs() < 1.0e-12);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "run,nx,nv,dt,impulse_amplitude,step,time,pulse_echo_real,pulse_echo_imag,pulse_echo_amplitude,control_echo_real,control_echo_imag,control_echo_amplitude,delta_echo_real,delta_echo_imag,delta_echo_amplitude,pulse_second_real,pulse_second_imag,pulse_second_amplitude,control_second_real,control_second_imag,control_second_amplitude,pulse_mass,pulse_momentum,pulse_kinetic_energy,pulse_field_energy,pulse_total_energy,pulse_poisson_residual,pulse_minimum_f,pulse_velocity_boundary_ratio,pulse_x_high_tail_fraction,pulse_v_high_tail_fraction,pulse_x_nyquist_fraction,pulse_v_nyquist_fraction,control_mass,control_momentum,control_total_energy,control_field_energy,control_poisson_residual,control_minimum_f,control_velocity_boundary_ratio,diagnostic_time"
    );

    println!(
        "# negative_control_columns,run,step,time,wrong_pulse_mode,predicted_echo_mode,impulse_amplitude,complex_response_amplitude,complex_response_phase,wrong_echo_mode_amplitude,wrong_echo_rephasing_time,off_target_original_mode_peak,off_target_peak_time,instantaneous_off_target_amplitude"
    );

    println!(
        "# negative_summary_columns,run,impulse_amplitude,wrong_pulse_mode,predicted_echo_mode,predicted_rephasing_time,sampled_peak_time,peak_amplitude,peak_phase,detected,off_target_peak,off_target_peak_time"
    );

    println!(
        "# cascade_columns,run,step,time,pulse_C2,control_C2,pulse_C2_relative_drift,control_C2_relative_drift,pulse_mean_C2,pulse_f_strophy,pulse_f_strophy_fraction,pulse_mixing_fraction,pulse_unmixing_fraction,pulse_zero_axis_fraction,pulse_nyquist_fraction,pulse_parseval_relative,control_mean_C2,control_f_strophy,control_mixing_fraction,control_unmixing_fraction,control_zero_axis_fraction,control_nyquist_fraction,control_parseval_relative"
    );

    println!(
        "# ridge_columns,run,step,time,mode,pulse_k_mode_f_strophy,pulse_positive_s_fraction,pulse_negative_s_fraction,pulse_zero_s_fraction,pulse_nyquist_s_fraction,pulse_signed_velocity_mode_index_centroid,pulse_dominant_signed_velocity_mode_index,control_k_mode_f_strophy,control_positive_s_fraction,control_negative_s_fraction,control_zero_s_fraction,control_nyquist_s_fraction,control_signed_velocity_mode_index_centroid,control_dominant_signed_velocity_mode_index,response_k_mode_f_strophy,response_positive_s_fraction,response_negative_s_fraction,response_zero_s_fraction,response_nyquist_s_fraction,response_signed_velocity_mode_index_centroid,response_dominant_signed_velocity_mode_index"
    );

    println!(
        "# flux_columns,run,step,time,x_cutoff,v_cutoff,pulse_phi_x,pulse_phi_v,pulse_cumulative_phi_x,pulse_cumulative_phi_v,pulse_direct_loss_x,pulse_direct_loss_v,pulse_closure_x,pulse_closure_v,control_phi_x,control_phi_v,control_cumulative_phi_x,control_cumulative_phi_v,control_direct_loss_x,control_direct_loss_v,control_closure_x,control_closure_v"
    );

    let common = |nx, nv, velocity_limit| VlasovPoissonConfig {
        nx,
        nv,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -velocity_limit,
        velocity_max: velocity_limit,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    };

    // One-factor-at-a-time refinements isolate spatial grid, velocity grid and
    // timestep effects. The combined fine case checks their interaction.
    // All configurations emit at the same 0.02 physical-time cadence.
    run_resolution("base_amp_0p025", common(64, 256, 6.0), 0.01, 2, 0.025)?;
    run_resolution("base", common(64, 256, 6.0), 0.01, 2, 0.05)?;
    run_resolution("base_amp_0p10", common(64, 256, 6.0), 0.01, 2, 0.10)?;
    run_resolution("x_refined", common(128, 256, 6.0), 0.01, 2, 0.05)?;
    run_resolution("v_refined", common(64, 512, 6.0), 0.01, 2, 0.05)?;
    run_resolution("dt_refined", common(64, 256, 6.0), 0.005, 4, 0.05)?;
    run_resolution("joint_refined", common(128, 512, 6.0), 0.005, 4, 0.05)?;
    // Extend velocity support while holding velocity spacing fixed (Nv doubles too).
    run_resolution("domain_expanded_fixed_dv", common(64, 512, 12.0), 0.01, 2, 0.05)?;
    // Extend velocity support at fixed Nv; dv doubles and recurrence time halves.
    run_resolution("domain_expanded_fixed_nv", common(64, 256, 12.0), 0.01, 2, 0.05)?;
    Ok(())
}
