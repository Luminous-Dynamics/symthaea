use std::error::Error;
use std::f64::consts::TAU;

use plasma_vlasov_reference::echo_analysis::{EchoPulsePair, predict_two_pulse_echo};
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
    let mut coarse_control = VlasovPoisson1d1v::new(coarse_config, coarse_initial)?;
    let mut fine_pulse = VlasovPoisson1d1v::new(fine_config, fine_initial.clone())?;
    let mut fine_control = VlasovPoisson1d1v::new(fine_config, fine_initial.clone())?;
    let mut sponge_pulse = VlasovPoisson1d1v::new(coarse_config, coarse_initial.clone())?;
    let mut sponge_control = VlasovPoisson1d1v::new(coarse_config, coarse_initial.clone())?;

    let initial_c2 = sponge_pulse.quadratic_casimir_budget()?.total_casimir;
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
    for step in 0..=total_steps {
        if step == pulse_step {
            coarse_pulse.apply_velocity_impulse(&second_kick)?;
            fine_pulse.apply_velocity_impulse(&second_kick)?;
            sponge_pulse.apply_velocity_impulse(&second_kick)?;
        }

        let time = coarse_pulse.time();
        if step % SAMPLE_EVERY == 0 || step == pulse_step || step == total_steps {
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
    let sponge_final_c2 = sponge_pulse.quadratic_casimir_budget()?.total_casimir;
    let sponge_direct_c2_drop = initial_c2 - sponge_final_c2;
    let c2_closure_error =
        (sponge_removed_c2 - sponge_direct_c2_drop).abs() / initial_c2.abs().max(f64::MIN_POSITIVE);

    println!(
        "summary,case,nv,t_nyquist,t_recurrence,echo_mode,predicted_rephase_time,coarse_peak_time,coarse_amplitude,coarse_phase,coarse_fwhm,fine_peak_time,fine_amplitude,fine_phase,fine_fwhm,sponge_peak_time,sponge_amplitude,sponge_phase,sponge_fwhm,sponge_to_fine_amplitude_ratio,sponge_pulse_removed_C2_fraction,sponge_C2_closure_error,sponge_pulse_removed_f_strophy,sponge_control_removed_C2_fraction,sponge_control_removed_f_strophy,coarse_initial_E,fine_initial_E,sponge_initial_E"
    );
    println!(
        "summary,delayed_echo,{},{:.8},{:.8},{},{:.8},{:.8},{:.12e},{:.8},{:.8},{:.8},{:.12e},{:.8},{:.8},{:.8},{:.12e},{:.8},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
        coarse_config.nv,
        coarse_horizon.nyquist_encounter_time,
        coarse_horizon.discrete_recurrence_time,
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
        sponge_removed_c2 / initial_c2.abs().max(f64::MIN_POSITIVE),
        c2_closure_error,
        sponge_removed_f_strophy,
        sponge_control_removed_c2 / initial_c2.abs().max(f64::MIN_POSITIVE),
        sponge_control_removed_f_strophy,
        initial_field_coarse.amplitude(),
        initial_field_fine.amplitude(),
        initial_field_sponge.amplitude(),
    );
    Ok(())
}
