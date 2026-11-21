use std::error::Error;

use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

fn main() -> Result<(), Box<dyn Error>> {
    let config = VlasovPoissonConfig {
        nx: 64,
        nv: 128,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -6.0,
        velocity_max: 6.0,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    };
    let mode = 1;
    let initial = maxwellian_perturbation(config, 1.0e-3, mode, 1.0)?;
    let mut solver = VlasovPoisson1d1v::new(config, initial)?;
    let dt = 0.01;
    let steps = 1_000;

    println!(
        "step,time,electric_mode_real,electric_mode_imag,electric_mode_amplitude,electric_mode_phase,mass,momentum,total_energy,poisson_residual,velocity_boundary_ratio,x_high_tail_fraction,v_high_tail_fraction,x_nyquist_fraction,v_nyquist_fraction"
    );
    for step in 0..=steps {
        if step % 10 == 0 {
            let d = solver.diagnostics()?;
            let coefficient = solver.electric_mode_coefficient(mode)?;
            let tails = solver.spectral_tail_diagnostics(0.75)?;
            println!(
                "{},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                step,
                d.time,
                coefficient.real,
                coefficient.imaginary,
                coefficient.amplitude(),
                coefficient.phase(),
                d.total_mass,
                d.momentum,
                d.total_energy,
                d.max_poisson_residual,
                d.velocity_boundary_ratio,
                tails.x.high_frequency_fraction,
                tails.velocity.high_frequency_fraction,
                tails.x.nyquist_energy_fraction,
                tails.velocity.nyquist_energy_fraction
            );
        }
        if step < steps {
            solver.step(dt)?;
        }
    }
    Ok(())
}
