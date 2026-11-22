use std::error::Error;
use std::time::Instant;

use plasma_vlasov_reference::phase_space_cascade::JointPhaseSpaceSpectrum;
use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

const STEPS_TO_RECURRENCE: usize = 4096;
const NYQUIST_GUARD_FRACTION: f64 = 0.9;

fn target_mode_guardband_fraction(spectrum: &JointPhaseSpaceSpectrum, spatial_mode: usize) -> f64 {
    let negative_x_mode = (spectrum.nx - spatial_mode) % spectrum.nx;
    let mut total = 0.0;
    let mut guardband = 0.0;
    for x_mode in [spatial_mode, negative_x_mode] {
        for velocity_index in 0..spectrum.nv {
            let energy = spectrum.mode_energy(x_mode, velocity_index).unwrap_or(0.0);
            total += energy;
            let signed_velocity = spectrum
                .signed_velocity_mode(velocity_index)
                .unwrap_or_default()
                .unsigned_abs() as f64;
            let normalized = signed_velocity / (spectrum.nv / 2) as f64;
            if normalized >= NYQUIST_GUARD_FRACTION {
                guardband += energy;
            }
        }
    }
    guardband / total.max(f64::MIN_POSITIVE)
}

fn run_case(
    label: &str,
    config: VlasovPoissonConfig,
    start_fraction: f64,
    maximum_damping_rate: f64,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let spatial_mode = 1usize;
    let horizon = config.velocity_sampling_horizon(spatial_mode)?;
    let initial = maxwellian_perturbation(config, 1.0e-2, spatial_mode, 1.0)?;
    let mut periodic = VlasovPoisson1d1v::new(config, initial.clone())?;
    let mut sponge = VlasovPoisson1d1v::new(config, initial.clone())?;
    let initial_field = periodic.electric_mode_coefficient(spatial_mode)?;
    let initial_budget = sponge.quadratic_casimir_budget()?;
    let initial_casimir = initial_budget.total_casimir;
    let dt = horizon.discrete_recurrence_time / STEPS_TO_RECURRENCE as f64;
    let mut removed_total_casimir = 0.0;
    let mut removed_fluctuation_f_strophy = 0.0;
    let mut periodic_nyquist_fraction = 0.0;
    let mut sponge_nyquist_fraction = 0.0;
    let mut periodic_guardband_fraction = 0.0;
    let mut sponge_guardband_fraction = 0.0;
    let mut periodic_nyquist_field_ratio = 0.0;
    let mut sponge_nyquist_field_ratio = 0.0;

    for step in 1..=STEPS_TO_RECURRENCE {
        periodic.advance_free_streaming(dt)?;

        let first_half = sponge.apply_velocity_fourier_sponge(
            0.5 * dt,
            start_fraction,
            maximum_damping_rate,
            2,
        )?;
        sponge.advance_free_streaming(dt)?;
        let second_half = sponge.apply_velocity_fourier_sponge(
            0.5 * dt,
            start_fraction,
            maximum_damping_rate,
            2,
        )?;
        removed_total_casimir +=
            first_half.removed_total_casimir + second_half.removed_total_casimir;
        removed_fluctuation_f_strophy +=
            first_half.removed_fluctuation_f_strophy + second_half.removed_fluctuation_f_strophy;

        if step == STEPS_TO_RECURRENCE / 2 {
            let periodic_spectrum = periodic.joint_phase_space_spectrum()?;
            let sponge_spectrum = sponge.joint_phase_space_spectrum()?;
            periodic_nyquist_fraction = periodic_spectrum.nyquist_f_strophy
                / periodic_spectrum.spectral_f_strophy.max(f64::MIN_POSITIVE);
            sponge_nyquist_fraction = sponge_spectrum.nyquist_f_strophy
                / sponge_spectrum.spectral_f_strophy.max(f64::MIN_POSITIVE);
            periodic_guardband_fraction =
                target_mode_guardband_fraction(&periodic_spectrum, spatial_mode);
            sponge_guardband_fraction =
                target_mode_guardband_fraction(&sponge_spectrum, spatial_mode);
            periodic_nyquist_field_ratio = periodic
                .electric_mode_coefficient(spatial_mode)?
                .amplitude()
                / initial_field.amplitude().max(f64::MIN_POSITIVE);
            sponge_nyquist_field_ratio =
                sponge.electric_mode_coefficient(spatial_mode)?.amplitude()
                    / initial_field.amplitude().max(f64::MIN_POSITIVE);
        }
    }

    let periodic_distribution_error = periodic
        .distribution()
        .iter()
        .zip(initial.iter())
        .map(|(observed, expected)| (observed - expected).abs())
        .fold(0.0_f64, f64::max)
        / initial
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
    let sponge_distribution_error = sponge
        .distribution()
        .iter()
        .zip(initial.iter())
        .map(|(observed, expected)| (observed - expected).abs())
        .fold(0.0_f64, f64::max)
        / initial
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);

    let periodic_final_field = periodic.electric_mode_coefficient(spatial_mode)?;
    let sponge_final_field = sponge.electric_mode_coefficient(spatial_mode)?;
    let periodic_field_return_error = (periodic_final_field.real - initial_field.real)
        .hypot(periodic_final_field.imaginary - initial_field.imaginary)
        / initial_field.amplitude().max(f64::MIN_POSITIVE);
    let sponge_field_return_error = (sponge_final_field.real - initial_field.real)
        .hypot(sponge_final_field.imaginary - initial_field.imaginary)
        / initial_field.amplitude().max(f64::MIN_POSITIVE);
    let final_budget = sponge.quadratic_casimir_budget()?;
    let direct_casimir_drop = initial_casimir - final_budget.total_casimir;
    let casimir_accounting_error = (removed_total_casimir - direct_casimir_drop).abs()
        / initial_casimir.max(f64::MIN_POSITIVE);
    let removed_casimir_fraction = removed_total_casimir / initial_casimir.max(f64::MIN_POSITIVE);
    let direct_casimir_drop_fraction = direct_casimir_drop / initial_casimir.max(f64::MIN_POSITIVE);
    let final_f_strophy_fraction = final_budget.fluctuation_f_strophy
        / initial_budget.fluctuation_f_strophy.max(f64::MIN_POSITIVE);
    let nx = config.nx;
    let nv = config.nv;
    let velocity_spacing = horizon.velocity_spacing;
    let nyquist_time = horizon.nyquist_encounter_time;
    let recurrence_time = horizon.discrete_recurrence_time;
    let wall_seconds = started.elapsed().as_secs_f64();

    println!(
        "{label},{nx},{nv},{velocity_spacing:.8},{nyquist_time:.8},{recurrence_time:.8},{start_fraction:.8},{maximum_damping_rate:.8},{periodic_nyquist_fraction:.12e},{sponge_nyquist_fraction:.12e},{periodic_guardband_fraction:.12e},{sponge_guardband_fraction:.12e},{periodic_nyquist_field_ratio:.12e},{sponge_nyquist_field_ratio:.12e},{periodic_distribution_error:.12e},{sponge_distribution_error:.12e},{periodic_field_return_error:.12e},{sponge_field_return_error:.12e},{removed_casimir_fraction:.12e},{direct_casimir_drop_fraction:.12e},{casimir_accounting_error:.12e},{removed_fluctuation_f_strophy:.12e},{final_f_strophy_fraction:.12e},{wall_seconds:.6}"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "case,nx,nv,dv,t_nyquist,t_recurrence,sponge_start_fraction,max_damping_rate,periodic_global_nyquist_fraction,sponge_global_nyquist_fraction,periodic_target_guardband_fraction,sponge_target_guardband_fraction,periodic_E_ratio_at_tN,sponge_E_ratio_at_tN,periodic_distribution_return_error,sponge_distribution_return_error,periodic_E_return_error,sponge_E_return_error,cumulative_removed_C2_fraction,direct_C2_drop_fraction,C2_accounting_error_fraction,accumulated_sponge_f_strophy_loss,final_f_strophy_fraction,wall_seconds"
    );

    let common = VlasovPoissonConfig {
        nx: 64,
        nv: 256,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -6.0,
        velocity_max: 6.0,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    };

    run_case("mild", common, 0.90, 0.5)?;
    run_case("medium", common, 0.75, 0.5)?;
    run_case("strong", common, 0.60, 0.5)?;
    Ok(())
}
