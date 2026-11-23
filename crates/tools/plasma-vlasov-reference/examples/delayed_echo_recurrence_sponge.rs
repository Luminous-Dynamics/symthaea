use std::error::Error;
use std::f64::consts::TAU;

use plasma_vlasov_reference::echo_analysis::{EchoPulsePair, predict_two_pulse_echo};
use plasma_vlasov_reference::phase_space_cascade::JointPhaseSpaceSpectrum;
use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

const DT: f64 = 0.05;
const SECOND_PULSE_TIME: f64 = 80.0;
const FINAL_TIME: f64 = 170.0;
const SAMPLE_EVERY: usize = 2;
const FIRST_MODE: usize = 1;
const SECOND_MODE: usize = 2;
const INITIAL_AMPLITUDE: f64 = 1.0e-3;
const IMPULSE_AMPLITUDE: f64 = 0.10;
const SPONGE_START: f64 = 0.75;
const SPONGE_DAMPING: f64 = 0.5;

#[derive(Debug, Clone, Copy)]
struct Peak {
    time: f64,
    amplitude: f64,
    phase: f64,
    fwhm: f64,
}

fn interpolate_crossing(
    left_time: f64,
    left_amplitude: f64,
    right_time: f64,
    right_amplitude: f64,
    target: f64,
) -> f64 {
    let denominator = right_amplitude - left_amplitude;
    if denominator.abs() <= f64::MIN_POSITIVE {
        return 0.5 * (left_time + right_time);
    }
    left_time + (target - left_amplitude) * (right_time - left_time) / denominator
}

fn measure_peak(samples: &[(f64, f64, f64)]) -> Peak {
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
    let peak_amplitude = amplitudes.get(peak_index).copied().unwrap_or(0.0);
    let half = 0.5 * peak_amplitude;

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
    let (time, real, imaginary) = samples.get(peak_index).copied().unwrap_or((0.0, 0.0, 0.0));
    Peak {
        time,
        amplitude: peak_amplitude,
        phase: imaginary.atan2(real),
        fwhm,
    }
}

#[derive(Debug, Clone, Copy)]
struct ResponseModeGuardbandMetrics {
    total_f_strophy: f64,
    guard_075_f_strophy: f64,
    guard_090_f_strophy: f64,
    nyquist_f_strophy: f64,
    guard_075_fraction: f64,
    guard_090_fraction: f64,
    nyquist_fraction: f64,
    signed_velocity_mode_centroid: f64,
}

fn response_mode_guardband_metrics(
    config: VlasovPoissonConfig,
    pulse: &VlasovPoisson1d1v,
    control: &VlasovPoisson1d1v,
    spatial_mode: usize,
) -> Result<ResponseModeGuardbandMetrics, Box<dyn Error>> {
    let response_distribution: Vec<f64> = pulse
        .distribution()
        .iter()
        .zip(control.distribution())
        .map(|(pulse_value, control_value)| pulse_value - control_value)
        .collect();
    let response_state = VlasovPoisson1d1v::new(config, response_distribution)?;
    let spectrum: JointPhaseSpaceSpectrum = response_state.joint_phase_space_spectrum()?;

    // Measure only the registered positive-k branch. Adding its -k
    // conjugate would cancel the signed-s centroid identically for real f.
    let mut total = 0.0;
    let mut guard_075 = 0.0;
    let mut guard_090 = 0.0;
    let mut nyquist = 0.0;
    let mut signed_s_moment = 0.0;
    let mut non_nyquist_total = 0.0;

    for velocity_index in 0..spectrum.nv {
        let energy = spectrum
            .mode_energy(spatial_mode, velocity_index)
            .unwrap_or(0.0);
        total += energy;
        if velocity_index == spectrum.nv / 2 {
            nyquist += energy;
            continue;
        }

        let signed_s = spectrum
            .signed_velocity_mode(velocity_index)
            .unwrap_or_default();
        let normalized = signed_s.unsigned_abs() as f64 / (spectrum.nv / 2) as f64;
        if normalized >= 0.75 {
            guard_075 += energy;
        }
        if normalized >= 0.90 {
            guard_090 += energy;
        }
        signed_s_moment += signed_s as f64 * energy;
        non_nyquist_total += energy;
    }

    let denominator = total.max(f64::MIN_POSITIVE);
    let centroid = if non_nyquist_total > 0.0 {
        signed_s_moment / non_nyquist_total
    } else {
        0.0
    };
    Ok(ResponseModeGuardbandMetrics {
        total_f_strophy: total,
        guard_075_f_strophy: guard_075,
        guard_090_f_strophy: guard_090,
        nyquist_f_strophy: nyquist,
        guard_075_fraction: guard_075 / denominator,
        guard_090_fraction: guard_090 / denominator,
        nyquist_fraction: nyquist / denominator,
        signed_velocity_mode_centroid: centroid,
    })
}

#[derive(Debug, Clone, Copy)]
struct PositivityMetrics {
    minimum: f64,
    negative_cell_fraction: f64,
    negative_mass_fraction: f64,
}

fn positivity_metrics(distribution: &[f64], config: VlasovPoissonConfig) -> PositivityMetrics {
    let mut minimum = f64::INFINITY;
    let mut negative_cells = 0usize;
    let mut positive_mass = 0.0;
    let mut negative_mass = 0.0;
    let cell_measure = config.dx() * config.dv();
    for value in distribution.iter().copied() {
        minimum = minimum.min(value);
        if value < 0.0 {
            negative_cells += 1;
            negative_mass += -value * cell_measure;
        } else {
            positive_mass += value * cell_measure;
        }
    }
    PositivityMetrics {
        minimum,
        negative_cell_fraction: negative_cells as f64 / distribution.len() as f64,
        negative_mass_fraction: negative_mass / positive_mass.max(f64::MIN_POSITIVE),
    }
}

fn relative_delta(after: f64, before: f64) -> f64 {
    (after - before) / before.abs().max(f64::MIN_POSITIVE)
}

fn config(nv: usize) -> VlasovPoissonConfig {
    VlasovPoissonConfig {
        nx: 64,
        nv,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -6.0,
        velocity_max: 6.0,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let coarse_config = config(256);
    let fine_config = config(512);
    let pulse_pair = EchoPulsePair {
        first_mode: FIRST_MODE as i32,
        second_mode: SECOND_MODE as i32,
        first_time: 0.0,
        second_time: SECOND_PULSE_TIME,
    };
    let prediction = predict_two_pulse_echo(pulse_pair)?;

    let coarse_horizon = coarse_config.velocity_sampling_horizon(FIRST_MODE)?;
    let fine_horizon = fine_config.velocity_sampling_horizon(FIRST_MODE)?;
    let pulse_step = (SECOND_PULSE_TIME / DT).round() as usize;
    let total_steps = (FINAL_TIME / DT).round() as usize;
    if (pulse_step as f64 * DT - SECOND_PULSE_TIME).abs() > 1.0e-12
        || (total_steps as f64 * DT - FINAL_TIME).abs() > 1.0e-12
    {
        return Err("pulse and final times must lie on the declared timestep grid".into());
    }

    let coarse_initial =
        maxwellian_perturbation(coarse_config, INITIAL_AMPLITUDE, FIRST_MODE, 1.0)?;
    let fine_initial = maxwellian_perturbation(fine_config, INITIAL_AMPLITUDE, FIRST_MODE, 1.0)?;
    let mut coarse_pulse = VlasovPoisson1d1v::new(coarse_config, coarse_initial.clone())?;
    let mut coarse_control = VlasovPoisson1d1v::new(coarse_config, coarse_initial.clone())?;
    let mut fine_pulse = VlasovPoisson1d1v::new(fine_config, fine_initial.clone())?;
    let mut fine_control = VlasovPoisson1d1v::new(fine_config, fine_initial.clone())?;
    let mut sponge_pulse = VlasovPoisson1d1v::new(coarse_config, coarse_initial.clone())?;
    let mut sponge_control = VlasovPoisson1d1v::new(coarse_config, coarse_initial.clone())?;

    let coarse_initial_budget = coarse_pulse.quadratic_casimir_budget()?;
    let fine_initial_budget = fine_pulse.quadratic_casimir_budget()?;
    let sponge_initial_budget = sponge_pulse.quadratic_casimir_budget()?;
    let coarse_initial_diagnostics = coarse_pulse.diagnostics()?;
    let fine_initial_diagnostics = fine_pulse.diagnostics()?;
    let sponge_initial_diagnostics = sponge_pulse.diagnostics()?;
    let coarse_initial_c2 = coarse_initial_budget.total_casimir;
    let fine_initial_c2 = fine_initial_budget.total_casimir;
    let initial_c2 = sponge_initial_budget.total_casimir;
    let mut sponge_removed_c2 = 0.0;
    let mut sponge_control_removed_c2 = 0.0;
    let mut sponge_removed_f_strophy = 0.0;
    let mut sponge_control_removed_f_strophy = 0.0;
    let initial_field_coarse =
        coarse_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
    let initial_field_fine = fine_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
    let initial_field_sponge =
        sponge_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;

    let second_kick: Vec<f64> = (0..coarse_config.nx)
        .map(|ix| {
            let phase =
                TAU * SECOND_MODE as f64 * coarse_config.x(ix) / coarse_config.domain_length;
            IMPULSE_AMPLITUDE * phase.sin()
        })
        .collect();

    let mut coarse_trace = Vec::new();
    let mut fine_trace = Vec::new();
    let mut sponge_trace = Vec::new();

    println!(
        "trace_columns,time,coarse_real,coarse_imag,fine_real,fine_imag,sponge_real,sponge_imag"
    );
    println!(
        "response_spectrum_columns,time,coarse_total_f_strophy,coarse_guard075_fraction,coarse_guard075_abs,coarse_guard090_fraction,coarse_guard090_abs,coarse_nyquist_fraction,coarse_nyquist_abs,coarse_signed_s_centroid,fine_total_f_strophy,fine_guard075_fraction,fine_guard075_abs,fine_guard090_fraction,fine_guard090_abs,fine_nyquist_fraction,fine_nyquist_abs,fine_signed_s_centroid,sponge_total_f_strophy,sponge_guard075_fraction,sponge_guard075_abs,sponge_guard090_fraction,sponge_guard090_abs,sponge_nyquist_fraction,sponge_nyquist_abs,sponge_signed_s_centroid"
    );
    for step in 0..=total_steps {
        if step == pulse_step {
            coarse_pulse.apply_velocity_impulse(&second_kick)?;
            fine_pulse.apply_velocity_impulse(&second_kick)?;
            sponge_pulse.apply_velocity_impulse(&second_kick)?;
        }

        let time = coarse_pulse.time();
        if step % SAMPLE_EVERY == 0 || step == pulse_step || step == total_steps {
            if step % (SAMPLE_EVERY * 10) == 0
                && (prediction.echo_time - 6.0..=FINAL_TIME).contains(&time)
            {
                let coarse_metrics = response_mode_guardband_metrics(
                    coarse_config,
                    &coarse_pulse,
                    &coarse_control,
                    prediction.echo_mode as usize,
                )?;
                let fine_metrics = response_mode_guardband_metrics(
                    fine_config,
                    &fine_pulse,
                    &fine_control,
                    prediction.echo_mode as usize,
                )?;
                let sponge_metrics = response_mode_guardband_metrics(
                    coarse_config,
                    &sponge_pulse,
                    &sponge_control,
                    prediction.echo_mode as usize,
                )?;
                println!(
                    "response_spectrum,{time:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.8}",
                    coarse_metrics.total_f_strophy,
                    coarse_metrics.guard_075_fraction,
                    coarse_metrics.guard_075_f_strophy,
                    coarse_metrics.guard_090_fraction,
                    coarse_metrics.guard_090_f_strophy,
                    coarse_metrics.nyquist_fraction,
                    coarse_metrics.nyquist_f_strophy,
                    coarse_metrics.signed_velocity_mode_centroid,
                    fine_metrics.total_f_strophy,
                    fine_metrics.guard_075_fraction,
                    fine_metrics.guard_075_f_strophy,
                    fine_metrics.guard_090_fraction,
                    fine_metrics.guard_090_f_strophy,
                    fine_metrics.nyquist_fraction,
                    fine_metrics.nyquist_f_strophy,
                    fine_metrics.signed_velocity_mode_centroid,
                    sponge_metrics.total_f_strophy,
                    sponge_metrics.guard_075_fraction,
                    sponge_metrics.guard_075_f_strophy,
                    sponge_metrics.guard_090_fraction,
                    sponge_metrics.guard_090_f_strophy,
                    sponge_metrics.nyquist_fraction,
                    sponge_metrics.nyquist_f_strophy,
                    sponge_metrics.signed_velocity_mode_centroid,
                );
            }

            let coarse_pulse_mode =
                coarse_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let coarse_control_mode =
                coarse_control.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let fine_pulse_mode =
                fine_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let fine_control_mode =
                fine_control.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let sponge_pulse_mode =
                sponge_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let sponge_control_mode =
                sponge_control.electric_mode_coefficient(prediction.echo_mode as usize)?;

            let coarse_response = (
                time,
                coarse_pulse_mode.real - coarse_control_mode.real,
                coarse_pulse_mode.imaginary - coarse_control_mode.imaginary,
            );
            let fine_response = (
                time,
                fine_pulse_mode.real - fine_control_mode.real,
                fine_pulse_mode.imaginary - fine_control_mode.imaginary,
            );
            let sponge_response = (
                time,
                sponge_pulse_mode.real - sponge_control_mode.real,
                sponge_pulse_mode.imaginary - sponge_control_mode.imaginary,
            );
            if (prediction.echo_time - 6.0..=FINAL_TIME).contains(&time) {
                coarse_trace.push(coarse_response);
                fine_trace.push(fine_response);
                sponge_trace.push(sponge_response);
                println!(
                    "trace,{time:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                    coarse_response.1,
                    coarse_response.2,
                    fine_response.1,
                    fine_response.2,
                    sponge_response.1,
                    sponge_response.2,
                );
            }
        }

        if step < total_steps {
            coarse_pulse.step(DT)?;
            coarse_control.step(DT)?;
            fine_pulse.step(DT)?;
            fine_control.step(DT)?;

            let first_half = sponge_pulse.apply_velocity_fourier_sponge(
                DT * 0.5,
                SPONGE_START,
                SPONGE_DAMPING,
                2,
            )?;
            let control_first_half = sponge_control.apply_velocity_fourier_sponge(
                DT * 0.5,
                SPONGE_START,
                SPONGE_DAMPING,
                2,
            )?;
            sponge_removed_c2 += first_half.removed_total_casimir;
            sponge_control_removed_c2 += control_first_half.removed_total_casimir;
            sponge_removed_f_strophy += first_half.removed_fluctuation_f_strophy;
            sponge_control_removed_f_strophy += control_first_half.removed_fluctuation_f_strophy;
            sponge_pulse.step(DT)?;
            sponge_control.step(DT)?;
            let second_half = sponge_pulse.apply_velocity_fourier_sponge(
                DT * 0.5,
                SPONGE_START,
                SPONGE_DAMPING,
                2,
            )?;
            let control_second_half = sponge_control.apply_velocity_fourier_sponge(
                DT * 0.5,
                SPONGE_START,
                SPONGE_DAMPING,
                2,
            )?;
            sponge_removed_c2 += second_half.removed_total_casimir;
            sponge_control_removed_c2 += control_second_half.removed_total_casimir;
            sponge_removed_f_strophy += second_half.removed_fluctuation_f_strophy;
            sponge_control_removed_f_strophy += control_second_half.removed_fluctuation_f_strophy;
        }
    }

    let coarse_peak = measure_peak(&coarse_trace);
    let fine_peak = measure_peak(&fine_trace);
    let sponge_peak = measure_peak(&sponge_trace);
    let coarse_final_budget = coarse_pulse.quadratic_casimir_budget()?;
    let fine_final_budget = fine_pulse.quadratic_casimir_budget()?;
    let sponge_final_budget = sponge_pulse.quadratic_casimir_budget()?;
    let coarse_final_diagnostics = coarse_pulse.diagnostics()?;
    let fine_final_diagnostics = fine_pulse.diagnostics()?;
    let sponge_final_diagnostics = sponge_pulse.diagnostics()?;
    let sponge_control_final_diagnostics = sponge_control.diagnostics()?;
    let sponge_direct_c2_drop = initial_c2 - sponge_final_budget.total_casimir;
    let coarse_positivity = positivity_metrics(coarse_pulse.distribution(), coarse_config);
    let fine_positivity = positivity_metrics(fine_pulse.distribution(), fine_config);
    let sponge_positivity = positivity_metrics(sponge_pulse.distribution(), coarse_config);

    let coarse_c2_relative_delta =
        relative_delta(coarse_final_budget.total_casimir, coarse_initial_c2);
    let fine_c2_relative_delta = relative_delta(fine_final_budget.total_casimir, fine_initial_c2);
    let sponge_c2_relative_delta = relative_delta(sponge_final_budget.total_casimir, initial_c2);

    let coarse_total_energy_relative_delta = relative_delta(
        coarse_final_diagnostics.total_energy,
        coarse_initial_diagnostics.total_energy,
    );
    let fine_total_energy_relative_delta = relative_delta(
        fine_final_diagnostics.total_energy,
        fine_initial_diagnostics.total_energy,
    );
    let sponge_total_energy_relative_delta = relative_delta(
        sponge_final_diagnostics.total_energy,
        sponge_initial_diagnostics.total_energy,
    );
    let coarse_kinetic_energy_relative_delta = relative_delta(
        coarse_final_diagnostics.kinetic_energy,
        coarse_initial_diagnostics.kinetic_energy,
    );
    let fine_kinetic_energy_relative_delta = relative_delta(
        fine_final_diagnostics.kinetic_energy,
        fine_initial_diagnostics.kinetic_energy,
    );
    let sponge_kinetic_energy_relative_delta = relative_delta(
        sponge_final_diagnostics.kinetic_energy,
        sponge_initial_diagnostics.kinetic_energy,
    );
    let coarse_field_energy_relative_delta = relative_delta(
        coarse_final_diagnostics.field_energy,
        coarse_initial_diagnostics.field_energy,
    );
    let fine_field_energy_relative_delta = relative_delta(
        fine_final_diagnostics.field_energy,
        fine_initial_diagnostics.field_energy,
    );
    let sponge_field_energy_relative_delta = relative_delta(
        sponge_final_diagnostics.field_energy,
        sponge_initial_diagnostics.field_energy,
    );
    let coarse_mass_relative_delta = relative_delta(
        coarse_final_diagnostics.total_mass,
        coarse_initial_diagnostics.total_mass,
    );
    let fine_mass_relative_delta = relative_delta(
        fine_final_diagnostics.total_mass,
        fine_initial_diagnostics.total_mass,
    );
    let sponge_mass_relative_delta = relative_delta(
        sponge_final_diagnostics.total_mass,
        sponge_initial_diagnostics.total_mass,
    );
    let coarse_momentum_delta =
        coarse_final_diagnostics.momentum - coarse_initial_diagnostics.momentum;
    let fine_momentum_delta = fine_final_diagnostics.momentum - fine_initial_diagnostics.momentum;
    let sponge_momentum_delta =
        sponge_final_diagnostics.momentum - sponge_initial_diagnostics.momentum;
    let c2_closure_error =
        (sponge_removed_c2 - sponge_direct_c2_drop).abs() / initial_c2.abs().max(f64::MIN_POSITIVE);
    let sponge_phase_error = (sponge_peak.phase - fine_peak.phase + std::f64::consts::PI)
        .rem_euclid(TAU)
        - std::f64::consts::PI;
    let sponge_fwhm_relative_error =
        (sponge_peak.fwhm - fine_peak.fwhm) / fine_peak.fwhm.abs().max(f64::MIN_POSITIVE);
    let sponge_amplitude_relative_error = (sponge_peak.amplitude - fine_peak.amplitude)
        / fine_peak.amplitude.abs().max(f64::MIN_POSITIVE);
    let sponge_minimum_f = sponge_final_diagnostics.minimum_distribution;
    let sponge_boundary_ratio = sponge_final_diagnostics.velocity_boundary_ratio;
    let control_total_energy_relative_delta = relative_delta(
        sponge_control_final_diagnostics.total_energy,
        sponge_initial_diagnostics.total_energy,
    );

    println!(
        "summary,case,nv,coarse_t_nyquist,coarse_t_recurrence,fine_t_nyquist,fine_t_recurrence,echo_mode,predicted_rephase_time,coarse_peak_time,coarse_amplitude,coarse_phase,coarse_fwhm,fine_peak_time,fine_amplitude,fine_phase,fine_fwhm,sponge_peak_time,sponge_amplitude,sponge_phase,sponge_fwhm,sponge_to_fine_amplitude_ratio,sponge_amplitude_relative_error,sponge_phase_error,sponge_peak_time_error,sponge_fwhm_relative_error,sponge_pulse_removed_C2_fraction,sponge_C2_closure_error,sponge_pulse_removed_f_strophy,sponge_control_removed_C2_fraction,sponge_control_removed_f_strophy,coarse_C2_relative_delta,fine_C2_relative_delta,sponge_C2_relative_delta,coarse_total_energy_relative_delta,fine_total_energy_relative_delta,sponge_total_energy_relative_delta,coarse_kinetic_energy_relative_delta,fine_kinetic_energy_relative_delta,sponge_kinetic_energy_relative_delta,coarse_field_energy_relative_delta,fine_field_energy_relative_delta,sponge_field_energy_relative_delta,coarse_mass_relative_delta,fine_mass_relative_delta,sponge_mass_relative_delta,coarse_momentum_delta,fine_momentum_delta,sponge_momentum_delta,coarse_minimum_f,coarse_negative_cell_fraction,coarse_negative_mass_fraction,fine_minimum_f,fine_negative_cell_fraction,fine_negative_mass_fraction,sponge_minimum_f,sponge_negative_cell_fraction,sponge_negative_mass_fraction,sponge_velocity_boundary_ratio,control_total_energy_relative_delta,coarse_initial_E,fine_initial_E,sponge_initial_E"
    );
    println!(
        "summary,delayed_echo,{},{:.8},{:.8},{:.8},{:.8},{},{:.8},{:.8},{:.12e},{:.8},{:.8},{:.8},{:.12e},{:.8},{:.8},{:.8},{:.12e},{:.8},{:.8},{:.12e},{:.12e},{:.12e},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}{:.12e}{:.12e}{:.12e}{:.12e}{:.12e}",
        coarse_config.nv,
        coarse_horizon.nyquist_encounter_time,
        coarse_horizon.discrete_recurrence_time,
        fine_horizon.nyquist_encounter_time,
        fine_horizon.discrete_recurrence_time,
        prediction.echo_mode,
        prediction.echo_time,
        coarse_peak.time,
        coarse_peak.amplitude,
        coarse_peak.phase,
        coarse_peak.fwhm,
        fine_peak.time,
        fine_peak.amplitude,
        fine_peak.phase,
        fine_peak.fwhm,
        sponge_peak.time,
        sponge_peak.amplitude,
        sponge_peak.phase,
        sponge_peak.fwhm,
        sponge_peak.amplitude / fine_peak.amplitude.max(f64::MIN_POSITIVE),
        sponge_amplitude_relative_error,
        sponge_phase_error,
        sponge_peak.time - fine_peak.time,
        sponge_fwhm_relative_error,
        sponge_removed_c2 / initial_c2.abs().max(f64::MIN_POSITIVE),
        c2_closure_error,
        sponge_removed_f_strophy,
        sponge_control_removed_c2 / initial_c2.abs().max(f64::MIN_POSITIVE),
        sponge_control_removed_f_strophy,
        coarse_c2_relative_delta,
        fine_c2_relative_delta,
        sponge_c2_relative_delta,
        coarse_total_energy_relative_delta,
        fine_total_energy_relative_delta,
        sponge_total_energy_relative_delta,
        coarse_kinetic_energy_relative_delta,
        fine_kinetic_energy_relative_delta,
        sponge_kinetic_energy_relative_delta,
        coarse_field_energy_relative_delta,
        fine_field_energy_relative_delta,
        sponge_field_energy_relative_delta,
        coarse_mass_relative_delta,
        fine_mass_relative_delta,
        sponge_mass_relative_delta,
        coarse_momentum_delta,
        fine_momentum_delta,
        sponge_momentum_delta,
        coarse_positivity.minimum,
        coarse_positivity.negative_cell_fraction,
        coarse_positivity.negative_mass_fraction,
        fine_positivity.minimum,
        fine_positivity.negative_cell_fraction,
        fine_positivity.negative_mass_fraction,
        sponge_positivity.minimum,
        sponge_positivity.negative_cell_fraction,
        sponge_positivity.negative_mass_fraction,
        sponge_boundary_ratio,
        control_total_energy_relative_delta,
        initial_field_coarse.amplitude(),
        initial_field_fine.amplitude(),
        initial_field_sponge.amplitude(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn synthetic_pair(
        velocity_mode_sign: f64,
    ) -> (VlasovPoissonConfig, VlasovPoisson1d1v, VlasovPoisson1d1v) {
        let config = VlasovPoissonConfig {
            nx: 32,
            nv: 128,
            domain_length: 2.0 * PI,
            velocity_min: -3.0,
            velocity_max: 3.0,
            charge_to_mass: -1.0,
            permittivity: 1.0,
            background_density: 1.0,
        };
        let base = maxwellian_perturbation(config, 0.0, 1, 1.0).unwrap();
        let mut perturbed = base.clone();
        let x_wave_number = TAU / config.domain_length;
        let v_wave_number = TAU * 8.0 / (config.velocity_max - config.velocity_min);
        for ix in 0..config.nx {
            for iv in 0..config.nv {
                let phase = x_wave_number * config.x(ix)
                    + velocity_mode_sign * v_wave_number * config.velocity(iv);
                perturbed[ix * config.nv + iv] += 1.0e-4 * phase.cos();
            }
        }
        let control = VlasovPoisson1d1v::new(config, base).unwrap();
        let pulse = VlasovPoisson1d1v::new(config, perturbed).unwrap();
        (config, pulse, control)
    }

    #[test]
    fn positive_k_response_ridge_keeps_positive_velocity_mode_sign() {
        let (config, pulse, control) = synthetic_pair(1.0);
        let metrics = response_mode_guardband_metrics(config, &pulse, &control, 1).unwrap();
        assert!(metrics.total_f_strophy > 0.0);
        assert!(metrics.signed_velocity_mode_centroid > 0.0);
        assert!(
            (metrics.signed_velocity_mode_centroid - 8.0).abs() < 1.0e-10,
            "positive-k branch centroid = {}",
            metrics.signed_velocity_mode_centroid
        );
        assert!(metrics.guard_075_f_strophy < metrics.total_f_strophy);
    }

    #[test]
    fn negative_k_response_ridge_keeps_negative_velocity_mode_sign() {
        let (config, pulse, control) = synthetic_pair(-1.0);
        let metrics = response_mode_guardband_metrics(config, &pulse, &control, 1).unwrap();
        assert!(metrics.total_f_strophy > 0.0);
        assert!(metrics.signed_velocity_mode_centroid < 0.0);
        assert!(
            (metrics.signed_velocity_mode_centroid + 8.0).abs() < 1.0e-10,
            "negative-k branch centroid = {}",
            metrics.signed_velocity_mode_centroid
        );
    }
}
