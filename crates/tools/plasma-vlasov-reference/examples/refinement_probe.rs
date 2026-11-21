use std::error::Error;
use plasma_vlasov_reference::{maxwellian_perturbation, VlasovPoisson1d1v, VlasovPoissonConfig};

fn run(nx: usize, nv: usize, dt: f64, final_time: f64) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let config = VlasovPoissonConfig {
        nx,
        nv,
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
    let steps = (final_time / dt).round() as usize;
    if (steps as f64 * dt - final_time).abs() > 1.0e-12 {
        return Err("final time must be an integer multiple of dt".into());
    }
    for _ in 0..steps {
        solver.step(dt)?;
    }
    let diagnostics = solver.diagnostics()?;
    Ok((
        solver.electric_mode_amplitude(mode)?,
        diagnostics.total_mass,
        diagnostics.max_poisson_residual,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let final_time = 1.0;
    println!("sweep,nx,nv,dt,final_time,electric_mode_1,mass,poisson_residual");

    for (nx, nv) in [(32, 64), (64, 128), (128, 256)] {
        let dt = 0.02 * 32.0 / nx as f64;
        let (amplitude, mass, residual) = run(nx, nv, dt, final_time)?;
        println!(
            "coupled_refinement,{nx},{nv},{dt:.8},{final_time:.8},{amplitude:.12e},{mass:.12e},{residual:.12e}"
        );
    }

    for dt in [0.04, 0.02, 0.01, 0.005] {
        let (amplitude, mass, residual) = run(64, 128, dt, final_time)?;
        println!(
            "time_refinement,64,128,{dt:.8},{final_time:.8},{amplitude:.12e},{mass:.12e},{residual:.12e}"
        );
    }
    Ok(())
}
