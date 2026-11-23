use std::error::Error;
use std::f64::consts::TAU;
use std::time::Instant;

use plasma_vlasov_reference::echo_analysis::{
    EchoMeasurementConfig, EchoModeTrace, EchoPulsePair, EchoTraceSample, measure_echo_trace,
};
use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

const NX: usize = 64;
const NV: usize = 256;
const DT: f64 = 0.01;
const FINAL_TIME: f64 = 12.0;
const SECOND_PULSE_TIME: f64 = 2.0;
const FIRST_MODE: usize = 1;
const SECOND_MODE: usize = 2;
const ECHO_SIGNAL_FLOOR: f64 = 1.0e-10;
const IMPULSE_AMPLITUDE: f64 = 0.05;

fn run_case(
    label: &str,
    start_fraction: f64,
    maximum_damping_rate: f64,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let config = VlasovPoissonConfig {
        nx: NX,
        nv: NV,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -6.0,
        velocity_max: 6.0,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    };
    let prediction =
        plasma_vlasov_reference::echo_analysis::predict_two_pulse_echo(EchoPulsePair {
            first_mode: FIRST_MODE as i32,
            second_mode: SECOND_MODE as i32,
            first_time: 0.0,
            second_time: SECOND_PULSE_TIME,
        })?;

    let initial = maxwellian_perturbation(config, 1.0e-3, FIRST_MODE, 1.0)?;
    let mut baseline_pulse = VlasovPoisson1d1v::new(config, initial.clone())?;
    let mut baseline_control = VlasovPoisson1d1v::new(config, initial.clone())?;
    let mut sponge_pulse = VlasovPoisson1d1v::new(config, initial.clone())?;
    let mut sponge_control = VlasovPoisson1d1v::new(config, initial)?;

    let kick: Vec<f64> = (0..config.nx)
        .map(|ix| {
            let phase = TAU * SECOND_MODE as f64 * config.x(ix) / config.domain_length;
            IMPULSE_AMPLITUDE * phase.sin()
        })
        .collect();

    let pulse_step = (SECOND_PULSE_TIME / DT).round() as usize;
    let total_steps = (FINAL_TIME / DT).round() as usize;
    if (pulse_step as f64 * DT - SECOND_PULSE_TIME).abs() > 1.0e-12
        || (total_steps as f64 * DT - FINAL_TIME).abs() > 1.0e-12
    {
        return Err("pulse and final times must lie on the fixed time grid".into());
    }

    let initial_budget = sponge_pulse.quadratic_casimir_budget()?;
    let initial_diagnostics = sponge_pulse.diagnostics()?;
    let mut pulse_removed_c2 = 0.0;
    let mut control_removed_c2 = 0.0;
    let mut pulse_removed_f_strophy = 0.0;
    let mut control_removed_f_strophy = 0.0;
    let mut baseline_trace = Vec::new();
    let mut sponge_trace = Vec::new();

    for step in 0..=total_steps {
        if step == pulse_step {
            baseline_pulse.apply_velocity_impulse(&kick)?;
            sponge_pulse.apply_velocity_impulse(&kick)?;
        }

        if step % 2 == 0 || step == pulse_step || step == total_steps {
            let time = baseline_pulse.time();
            let base_pulse =
                baseline_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let base_control =
                baseline_control.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let sponge_pulse_mode =
                sponge_pulse.electric_mode_coefficient(prediction.echo_mode as usize)?;
            let sponge_control_mode =
                sponge_control.electric_mode_coefficient(prediction.echo_mode as usize)?;

            baseline_trace.push(EchoTraceSample::new(
                time,
                base_pulse.real - base_control.real,
                base_pulse.imaginary - base_control.imaginary,
            ));
            sponge_trace.push(EchoTraceSample::new(
                time,
                sponge_pulse_mode.real - sponge_control_mode.real,
                sponge_pulse_mode.imaginary - sponge_control_mode.imaginary,
            ));
        }

        if step < total_steps {
            baseline_pulse.step(DT)?;
            baseline_control.step(DT)?;

            // Symmetric operator composition: sponge(dt/2), self-consistent
            // Vlasov-Poisson Strang step, sponge(dt/2). The sponge remains an
            // explicitly dissipative model with full loss accounting.
            let first_pulse_half = sponge_pulse.apply_velocity_fourier_sponge(
                DT * 0.5,
                start_fraction,
                maximum_damping_rate,
                2,
            )?;
            let first_control_half = sponge_control.apply_velocity_fourier_sponge(
                DT * 0.5,
                start_fraction,
                maximum_damping_rate,
                2,
            )?;
            pulse_removed_c2 += first_pulse_half.removed_total_casimir;
            pulse_removed_f_strophy += first_pulse_half.removed_fluctuation_f_strophy;
            control_removed_c2 += first_control_half.removed_total_casimir;
            control_removed_f_strophy += first_control_half.removed_fluctuation_f_strophy;

            sponge_pulse.step(DT)?;
            sponge_control.step(DT)?;

            let second_pulse_half = sponge_pulse.apply_velocity_fourier_sponge(
                DT * 0.5,
                start_fraction,
                maximum_damping_rate,
                2,
            )?;
            let second_control_half = sponge_control.apply_velocity_fourier_sponge(
                DT * 0.5,
                start_fraction,
                maximum_damping_rate,
                2,
            )?;
            pulse_removed_c2 += second_pulse_half.removed_total_casimir;
            pulse_removed_f_strophy += second_pulse_half.removed_fluctuation_f_strophy;
            control_removed_c2 += second_control_half.removed_total_casimir;
            control_removed_f_strophy += second_control_half.removed_fluctuation_f_strophy;
        }
    }

    let baseline_measurement = measure_echo_trace(
        prediction,
        EchoModeTrace {
            spatial_mode: prediction.echo_mode,
            samples: &baseline_trace,
        },
        EchoMeasurementConfig {
            minimum_peak_amplitude: ECHO_SIGNAL_FLOOR,
        },
    )?;
    let sponge_measurement = measure_echo_trace(
        prediction,
        EchoModeTrace {
            spatial_mode: prediction.echo_mode,
            samples: &sponge_trace,
        },
        EchoMeasurementConfig {
            minimum_peak_amplitude: ECHO_SIGNAL_FLOOR,
        },
    )?;

    let final_budget = sponge_pulse.quadratic_casimir_budget()?;
    let final_control_budget = sponge_control.quadratic_casimir_budget()?;
    let final_diagnostics = sponge_pulse.diagnostics()?;
    let direct_c2_drop = initial_budget.total_casimir - final_budget.total_casimir;
    let direct_control_c2_drop = initial_budget.total_casimir - final_control_budget.total_casimir;
    let cumulative_c2_fraction =
        pulse_removed_c2 / initial_budget.total_casimir.max(f64::MIN_POSITIVE);
    let direct_c2_fraction = direct_c2_drop / initial_budget.total_casimir.max(f64::MIN_POSITIVE);
    let c2_accounting_error = (pulse_removed_c2 - direct_c2_drop).abs()
        / initial_budget.total_casimir.max(f64::MIN_POSITIVE);
    let response_ratio = sponge_measurement.peak_amplitude
        / baseline_measurement.peak_amplitude.max(f64::MIN_POSITIVE);
    let mass_relative_delta = (final_diagnostics.total_mass - initial_diagnostics.total_mass)
        / initial_diagnostics.total_mass.abs().max(f64::MIN_POSITIVE);
    let kinetic_energy_relative_delta = (final_diagnostics.kinetic_energy
        - initial_diagnostics.kinetic_energy)
        / initial_diagnostics
            .kinetic_energy
            .abs()
            .max(f64::MIN_POSITIVE);
    let field_energy_relative_delta = (final_diagnostics.field_energy
        - initial_diagnostics.field_energy)
        / initial_diagnostics
            .field_energy
            .abs()
            .max(f64::MIN_POSITIVE);

    let control_cumulative_c2_fraction =
        control_removed_c2 / initial_budget.total_casimir.max(f64::MIN_POSITIVE);
    let control_direct_c2_drop_fraction =
        direct_control_c2_drop / initial_budget.total_casimir.max(f64::MIN_POSITIVE);
    let control_c2_accounting_error = (control_removed_c2 - direct_control_c2_drop).abs()
        / initial_budget.total_casimir.max(f64::MIN_POSITIVE);
    let echo_mode = prediction.echo_mode;
    let predicted_rephase_time = prediction.echo_time;
    let baseline_peak_amplitude = baseline_measurement.peak_amplitude;
    let sponge_peak_amplitude = sponge_measurement.peak_amplitude;
    let baseline_peak_time = baseline_measurement.sampled_peak_time;
    let sponge_peak_time = sponge_measurement.sampled_peak_time;
    let baseline_peak_phase = baseline_measurement.peak_phase_radians;
    let sponge_peak_phase = sponge_measurement.peak_phase_radians;
    let baseline_detected = baseline_measurement.detected;
    let sponge_detected = sponge_measurement.detected;
    let baseline_fwhm = baseline_measurement
        .full_width_half_maximum
        .unwrap_or(f64::NAN);
    let sponge_fwhm = sponge_measurement
        .full_width_half_maximum
        .unwrap_or(f64::NAN);
    let pulse_removed_f_strophy_value = pulse_removed_f_strophy;
    let control_removed_f_strophy_value = control_removed_f_strophy;
    let momentum_delta = final_diagnostics.momentum - initial_diagnostics.momentum;
    let minimum_f = final_diagnostics.minimum_distribution;
    let wall_seconds = started.elapsed().as_secs_f64();

    println!(
        "{label},{start_fraction:.4},{maximum_damping_rate:.4},{echo_mode},{predicted_rephase_time:.8},{baseline_peak_amplitude:.12e},{sponge_peak_amplitude:.12e},{baseline_peak_time:.8},{sponge_peak_time:.8},{response_ratio:.12e},{baseline_peak_phase:.12e},{sponge_peak_phase:.12e},{baseline_detected},{sponge_detected},{baseline_fwhm:.8},{sponge_fwhm:.8},{cumulative_c2_fraction:.12e},{direct_c2_fraction:.12e},{c2_accounting_error:.12e},{control_cumulative_c2_fraction:.12e},{control_direct_c2_drop_fraction:.12e},{control_c2_accounting_error:.12e},{pulse_removed_f_strophy_value:.12e},{control_removed_f_strophy_value:.12e},{mass_relative_delta:.12e},{momentum_delta:.12e},{kinetic_energy_relative_delta:.12e},{field_energy_relative_delta:.12e},{minimum_f:.12e},{wall_seconds:.6}"
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "case,start_fraction,max_damping_rate,echo_mode,predicted_rephase_time,baseline_peak_amplitude,sponge_peak_amplitude,baseline_peak_time,sponge_peak_time,amplitude_ratio,baseline_peak_phase,sponge_peak_phase,baseline_detected,sponge_detected,baseline_fwhm,sponge_fwhm,pulse_cumulative_removed_C2_fraction,pulse_direct_C2_drop_fraction,pulse_C2_accounting_error_fraction,control_cumulative_removed_C2_fraction,control_direct_C2_drop_fraction,control_C2_accounting_error_fraction,pulse_removed_f_strophy,control_removed_f_strophy,mass_relative_delta,momentum_delta,kinetic_energy_relative_delta,field_energy_relative_delta,minimum_f,wall_seconds"
    );

    run_case("mild", 0.90, 0.5)?;
    run_case("medium", 0.75, 0.5)?;
    run_case("strong", 0.60, 0.5)?;
    Ok(())
}
