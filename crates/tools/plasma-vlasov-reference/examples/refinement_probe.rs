use std::error::Error;

use plasma_vlasov_reference::{VlasovPoisson1d1v, VlasovPoissonConfig, maxwellian_perturbation};

#[derive(Debug, Clone, Copy)]
struct ProbeResult {
    electric_mode_1: f64,
    mass: f64,
    total_energy: f64,
    poisson_residual: f64,
    velocity_boundary_ratio: f64,
    minimum_distribution: f64,
    x_high_tail_fraction: f64,
    velocity_high_tail_fraction: f64,
}

fn run(
    nx: usize,
    nv: usize,
    velocity_max: f64,
    dt: f64,
    final_time: f64,
) -> Result<ProbeResult, Box<dyn Error>> {
    let config = VlasovPoissonConfig {
        nx,
        nv,
        domain_length: 4.0 * std::f64::consts::PI,
        velocity_min: -velocity_max,
        velocity_max,
        charge_to_mass: -1.0,
        permittivity: 1.0,
        background_density: 1.0,
    };
    let initial = maxwellian_perturbation(config, 1.0e-3, 1, 1.0)?;
    let mut solver = VlasovPoisson1d1v::new(config, initial)?;
    let steps = (final_time / dt).round() as usize;
    if (steps as f64 * dt - final_time).abs() > 1.0e-12 {
        return Err("final time must be an integer multiple of dt".into());
    }
    for _ in 0..steps {
        solver.step(dt)?;
    }
    let diagnostics = solver.diagnostics()?;
    let tails = solver.spectral_tail_diagnostics(0.75)?;
    Ok(ProbeResult {
        electric_mode_1: solver.electric_mode_amplitude(1)?,
        mass: diagnostics.total_mass,
        total_energy: diagnostics.total_energy,
        poisson_residual: diagnostics.max_poisson_residual,
        velocity_boundary_ratio: diagnostics.velocity_boundary_ratio,
        minimum_distribution: diagnostics.minimum_distribution,
        x_high_tail_fraction: tails.x.high_frequency_fraction,
        velocity_high_tail_fraction: tails.velocity.high_frequency_fraction,
    })
}

fn print_row(
    axis: &str,
    nx: usize,
    nv: usize,
    velocity_max: f64,
    dt: f64,
    final_time: f64,
    result: ProbeResult,
) {
    println!(
        "{axis},{nx},{nv},{velocity_max:.8},{dt:.8},{final_time:.8},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
        result.electric_mode_1,
        result.mass,
        result.total_energy,
        result.poisson_residual,
        result.velocity_boundary_ratio,
        result.minimum_distribution,
        result.x_high_tail_fraction,
        result.velocity_high_tail_fraction
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let final_time = 1.0;
    println!(
        "sweep,nx,nv,velocity_max,dt,final_time,electric_mode_1,mass,total_energy,poisson_residual,velocity_boundary_ratio,minimum_distribution,x_high_tail_fraction,v_high_tail_fraction"
    );

    // Spatial grid refinement with velocity resolution, domain and timestep fixed.
    for nx in [32, 64, 128] {
        let result = run(nx, 128, 6.0, 0.01, final_time)?;
        print_row("space", nx, 128, 6.0, 0.01, final_time, result);
    }

    // Velocity grid refinement with spatial resolution and domain fixed.
    for nv in [64, 128, 256] {
        let result = run(64, nv, 6.0, 0.01, final_time)?;
        print_row("velocity_resolution", 64, nv, 6.0, 0.01, final_time, result);
    }

    // Velocity-domain expansion at fixed spatial and velocity grid counts.
    for velocity_max in [4.0, 6.0, 8.0] {
        let result = run(64, 128, velocity_max, 0.01, final_time)?;
        print_row(
            "velocity_domain",
            64,
            128,
            velocity_max,
            0.01,
            final_time,
            result,
        );
    }

    // Independent timestep refinement at fixed phase-space grid and domain.
    for dt in [0.04, 0.02, 0.01, 0.005] {
        let result = run(64, 128, 6.0, dt, final_time)?;
        print_row("time", 64, 128, 6.0, dt, final_time, result);
    }
    Ok(())
}
