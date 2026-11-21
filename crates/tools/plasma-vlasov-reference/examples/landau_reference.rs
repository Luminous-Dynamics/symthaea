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
    let steps = 200;

    println!(
        "step,time,electric_mode_1,mass,momentum,total_energy,poisson_residual,velocity_boundary_ratio"
    );
    for step in 0..=steps {
        if step % 10 == 0 {
            let d = solver.diagnostics()?;
            println!(
                "{},{:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                step,
                d.time,
                solver.electric_mode_amplitude(mode)?,
                d.total_mass,
                d.momentum,
                d.total_energy,
                d.max_poisson_residual,
                d.velocity_boundary_ratio
            );
        }
        if step < steps {
            solver.step(dt)?;
        }
    }
    Ok(())
}
