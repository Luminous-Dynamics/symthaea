use std::error::Error;

use plasma_vlasov_reference::phase_space_cascade::JointPhaseSpaceSpectrum;
use plasma_vlasov_reference::{
    VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation,
};

const ALPHA: f64 = 0.05;
const MODE: usize = 1;
const FACTOR: usize = 2;
const GUARD_FRACTION: f64 = 0.9;

fn relative_linf_error(observed: &[f64], expected: &[f64]) -> f64 {
    let maximum_error = observed
        .iter()
        .zip(expected.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let scale = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    maximum_error / scale
}

fn guardband_fraction(
    spectrum: &JointPhaseSpaceSpectrum,
    spatial_mode: usize,
) -> f64 {
    let negative_mode = (spectrum.nx - spatial_mode) % spectrum.nx;
    let mut total = 0.0;
    let mut guardband = 0.0;
    for x_mode in [spatial_mode, negative_mode] {
        for velocity_index in 0..spectrum.nv {
            let energy = spectrum.mode_energy(x_mode, velocity_index).unwrap_or(0.0);
            total += energy;
            let signed = spectrum
                .signed_velocity_mode(velocity_index)
                .unwrap_or_default()
                .unsigned_abs() as f64;
            let normalized = signed / (spectrum.nv / 2) as f64;
            if normalized >= GUARD_FRACTION {
                guardband += energy;
            }
        }
    }
    guardband / total.max(f64::MIN_POSITIVE)
}

fn analytic_free_streaming_state(
    config: VlasovPoissonConfig,
    alpha: f64,
    mode: usize,
    time: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let initial = maxwellian_perturbation(config, alpha, mode, 1.0)?;
    let wave_number = std::f64::consts::TAU * mode as f64 / config.domain_length;
    let mut expected = vec![0.0; config.nx * config.nv];
    for ix in 0..config.nx {
        let x = config.x(ix);
        let initial_density = 1.0 + alpha * (wave_number * x).cos();
        for iv in 0..config.nv {
            let equilibrium = initial[ix * config.nv + iv] / initial_density;
            let phase = wave_number * (x - config.velocity(iv) * time);
            expected[ix * config.nv + iv] = equilibrium * (1.0 + alpha * phase.cos());
        }
    }
    Ok(expected)
}

fn main() -> Result<(), Box<dyn Error>> {
    let coarse_config = VlasovPoissonConfig {
        nx: 64,
        nv: 128,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -6.0,
        velocity_max: 6.0,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    };
    let old_horizon = coarse_config.velocity_sampling_horizon(MODE)?;
    let fine_config = VlasovPoissonConfig {
        nv: coarse_config.nv * FACTOR,
        ..coarse_config
    };
    let fine_horizon = fine_config.velocity_sampling_horizon(MODE)?;
    let initial = maxwellian_perturbation(coarse_config, ALPHA, MODE, 1.0)?;
    let initial_fine = maxwellian_perturbation(fine_config, ALPHA, MODE, 1.0)?;

    // Baseline: no adaptation, return at the coarse discrete recurrence.
    let mut coarse = VlasovPoisson1d1v::new(coarse_config, initial.clone())?;
    coarse.advance_free_streaming(old_horizon.discrete_recurrence_time)?;
    let coarse_recurrence_error = relative_linf_error(coarse.distribution(), &initial);

    // Early transition: refine at 40% of the old Nyquist encounter time.
    let early_refine_time = 0.4 * old_horizon.nyquist_encounter_time;
    let mut early = VlasovPoisson1d1v::new(coarse_config, initial.clone())?;
    early.advance_free_streaming(early_refine_time)?;
    let early_guardband = guardband_fraction(&early.joint_phase_space_spectrum()?, MODE);
    let early_report = early.refine_velocity_grid_fourier(FACTOR)?;
    let remaining_to_old_recurrence =
        old_horizon.discrete_recurrence_time - early_refine_time;
    early.advance_free_streaming(remaining_to_old_recurrence)?;
    let early_exact_at_old_recurrence = analytic_free_streaming_state(
        fine_config,
        ALPHA,
        MODE,
        old_horizon.discrete_recurrence_time,
    )?;
    let early_characteristic_error =
        relative_linf_error(early.distribution(), &early_exact_at_old_recurrence);
    let early_old_node_error = early
        .distribution()
        .iter()
        .step_by(FACTOR)
        .zip(initial.iter())
        .map(|(observed, expected)| (observed - expected).abs())
        .fold(0.0_f64, f64::max)
        / initial
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);

    // Continue to the new recurrence horizon. The fine grid should return to
    // its own initially interpolated state, not to the coarse recurrence time.
    early.advance_free_streaming(
        fine_horizon.discrete_recurrence_time - old_horizon.discrete_recurrence_time,
    )?;
    let early_fine_recurrence_error = relative_linf_error(early.distribution(), &initial_fine);

    // Negative control: wait until after the old Nyquist encounter before
    // zero-padding. It can reproduce the aliased old samples but cannot
    // reconstruct the continuous characteristic solution that was lost.
    let late_refine_time = 1.5 * old_horizon.nyquist_encounter_time;
    let mut late = VlasovPoisson1d1v::new(coarse_config, initial)?;
    late.advance_free_streaming(late_refine_time)?;
    let late_guardband = guardband_fraction(&late.joint_phase_space_spectrum()?, MODE);
    let late_report = late.refine_velocity_grid_fourier(FACTOR)?;
    let late_exact_state =
        analytic_free_streaming_state(fine_config, ALPHA, MODE, late_refine_time)?;
    let late_characteristic_error =
        relative_linf_error(late.distribution(), &late_exact_state);

    println!(
        "case,source_nv,destination_nv,old_t_nyquist,old_t_recurrence,new_t_nyquist,new_t_recurrence,transition_time,guardband_fraction,old_node_error,characteristic_error,fine_recurrence_error,c2_relative_delta,mass_relative_delta,old_bytes,new_bytes"
    );
    println!(
        "coarse_baseline,{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},0.0,0.0,0,{}",
        coarse_config.nv,
        coarse_config.nv,
        old_horizon.nyquist_encounter_time,
        old_horizon.discrete_recurrence_time,
        old_horizon.nyquist_encounter_time,
        old_horizon.discrete_recurrence_time,
        old_horizon.discrete_recurrence_time,
        0.0,
        0.0,
        coarse_recurrence_error,
        0.0,
        coarse_config.nx * coarse_config.nv * std::mem::size_of::<f64>(),
    );
    println!(
        "refine_before_guard,{},{},{:.8},{:.8},{:.8},{:.8},{early_refine_time:.8},{early_guardband:.12e},{:.12e},{early_characteristic_error:.12e},{early_fine_recurrence_error:.12e},{:.12e},{:.12e},{},{},",
        early_report.old_nv,
        early_report.new_nv,
        old_horizon.nyquist_encounter_time,
        old_horizon.discrete_recurrence_time,
        fine_horizon.nyquist_encounter_time,
        fine_horizon.discrete_recurrence_time,
        early_old_node_error,
        early_characteristic_error,
        early_fine_recurrence_error,
        early_report.total_casimir_relative_delta,
        early_report.mass_relative_delta,
        early_report.old_state_bytes,
        early_report.new_state_bytes,
    );
    println!(
        "refine_after_alias,{},{},{:.8},{:.8},{:.8},{:.8},{late_refine_time:.8},{late_guardband:.12e},{:.12e},{late_characteristic_error:.12e},NaN,{:.12e},{:.12e},{},{},",
        late_report.old_nv,
        late_report.new_nv,
        old_horizon.nyquist_encounter_time,
        old_horizon.discrete_recurrence_time,
        fine_horizon.nyquist_encounter_time,
        fine_horizon.discrete_recurrence_time,
        late_report.maximum_old_node_interpolation_error,
        late_characteristic_error,
        late_report.total_casimir_relative_delta,
        late_report.mass_relative_delta,
        late_report.old_state_bytes,
        late_report.new_state_bytes,
    );

    Ok(())
}
