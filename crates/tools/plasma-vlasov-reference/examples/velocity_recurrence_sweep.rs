use std::error::Error;
use std::time::Instant;

use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

fn target_mode_velocity_guardbands(
    spectrum: &plasma_vlasov_reference::phase_space_cascade::JointPhaseSpaceSpectrum,
    positive_x_mode: usize,
    guard_fraction: f64,
) -> (f64, f64) {
    let negative_x_mode = (spectrum.nx - positive_x_mode) % spectrum.nx;
    let mut branch_total = 0.0;
    let mut branch_nyquist = 0.0;
    let mut branch_guardband = 0.0;
    for x_mode in [positive_x_mode, negative_x_mode] {
        for velocity_index in 0..spectrum.nv {
            let energy = spectrum.mode_energy(x_mode, velocity_index).unwrap_or(0.0);
            branch_total += energy;
            let signed_velocity = spectrum
                .signed_velocity_mode(velocity_index)
                .unwrap_or_default()
                .unsigned_abs() as f64;
            let normalized = signed_velocity / (spectrum.nv / 2) as f64;
            if velocity_index == spectrum.nv / 2 {
                branch_nyquist += energy;
            }
            if normalized >= guard_fraction {
                branch_guardband += energy;
            }
        }
    }
    let denominator = branch_total.max(f64::MIN_POSITIVE);
    (branch_nyquist / denominator, branch_guardband / denominator)
}

fn run_case(
    label: &str,
    config: VlasovPoissonConfig,
    spatial_mode: usize,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let horizon = config.velocity_sampling_horizon(spatial_mode)?;
    let initial = maxwellian_perturbation(config, 1.0e-2, spatial_mode, 1.0)?;
    let mut solver = VlasovPoisson1d1v::new(config, initial.clone())?;
    let initial_field = solver.electric_mode_coefficient(spatial_mode)?;

    solver.advance_free_streaming(horizon.nyquist_encounter_time)?;
    let nyquist_spectrum = solver.joint_phase_space_spectrum()?;
    let nyquist_fraction = nyquist_spectrum.nyquist_f_strophy
        / nyquist_spectrum.spectral_f_strophy.max(f64::MIN_POSITIVE);
    // Report both the exact Nyquist-bin occupancy and a fixed-width
    // normalized guard band for the registered +/- spatial-mode pair.
    // A guard band is less sensitive to how many discrete bins sample
    // the same physical high-s region.
    let (target_mode_nyquist_fraction, target_mode_guardband_fraction) =
        target_mode_velocity_guardbands(&nyquist_spectrum, spatial_mode, 0.9);

    solver.advance_free_streaming(
        horizon.discrete_recurrence_time - horizon.nyquist_encounter_time,
    )?;
    let final_distribution = solver.distribution();
    let maximum_absolute_error = final_distribution
        .iter()
        .zip(initial.iter())
        .map(|(observed, expected)| (observed - expected).abs())
        .fold(0.0_f64, f64::max);
    let distribution_scale = initial
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let relative_distribution_return_error = maximum_absolute_error / distribution_scale;

    let final_field = solver.electric_mode_coefficient(spatial_mode)?;
    let field_return_error = (final_field.real - initial_field.real)
        .hypot(final_field.imaginary - initial_field.imaginary)
        / initial_field.amplitude().max(f64::MIN_POSITIVE);

    println!(
        "{label},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.6}",
        config.nx,
        config.nv,
        config.velocity_min,
        config.velocity_max,
        horizon.velocity_spacing,
        horizon.physical_wave_number,
        horizon.nyquist_encounter_time,
        horizon.discrete_recurrence_time,
        nyquist_fraction,
        target_mode_nyquist_fraction,
        target_mode_guardband_fraction,
        relative_distribution_return_error,
        field_return_error,
        started.elapsed().as_secs_f64(),
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "case,nx,nv,v_min,v_max,dv,k,t_nyquist,t_recurrence,global_nyquist_f_strophy_fraction,target_mode_nyquist_fraction,target_mode_guardband_fraction,distribution_return_linf_relative,electric_mode_return_relative,wall_seconds"
    );

    let common = |nx: usize, nv: usize, velocity_limit: f64| VlasovPoissonConfig {
        nx,
        nv,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -velocity_limit,
        velocity_max: velocity_limit,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    };

    run_case("base", common(64, 256, 6.0), 1)?;
    run_case("velocity_refined_fixed_support", common(64, 512, 6.0), 1)?;
    run_case("support_expanded_fixed_dv", common(64, 512, 12.0), 1)?;
    run_case("support_expanded_fixed_nv", common(64, 256, 12.0), 1)?;
    Ok(())
}
