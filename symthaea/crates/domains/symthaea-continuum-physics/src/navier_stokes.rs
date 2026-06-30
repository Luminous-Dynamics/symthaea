// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Navier-Stokes solver for incompressible fluid dynamics.
//!
//! ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f  (momentum)
//! ∇·u = 0  (incompressibility)
//!
//! Uses projection method (Chorin 1968): predict velocity, solve pressure
//! Poisson equation, correct velocity to be divergence-free.
//!
//! References:
//! - Chorin, A. J. (1968). Math. Comput. 22, 745.
//! - Stam, J. (1999). "Stable Fluids". SIGGRAPH.

/// 2D incompressible Navier-Stokes solver.
#[derive(Debug, Clone)]
pub struct NavierStokes2D {
    /// Velocity field u (x-component)
    pub u: Vec<f64>,
    /// Velocity field v (y-component)
    pub v: Vec<f64>,
    /// Pressure field
    pub p: Vec<f64>,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    /// Grid spacing
    pub dx: f64,
    /// Kinematic viscosity (m²/s)
    pub nu: f64,
    /// Fluid density (kg/m³)
    pub rho: f64,
    /// Time step
    pub dt: f64,
}

#[inline]
fn idx(i: usize, j: usize, ny: usize) -> usize {
    i * ny + j
}

impl NavierStokes2D {
    /// Create a new 2D fluid domain.
    pub fn new(nx: usize, ny: usize, dx: f64, nu: f64) -> Self {
        let dt = 0.1 * dx * dx / nu.max(1e-6); // CFL for diffusion (conservative)
        let size = nx * ny;
        Self {
            u: vec![0.0; size],
            v: vec![0.0; size],
            p: vec![0.0; size],
            nx,
            ny,
            dx,
            nu,
            rho: 1.0,
            dt,
        }
    }

    #[allow(dead_code)]
    fn idx(&self, i: usize, j: usize) -> usize {
        idx(i, j, self.ny)
    }

    /// Advance one time step using Chorin's projection method.
    pub fn step_once(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dt = self.dt;
        let nu = self.nu;

        // Step 1: Predict velocity (advection + diffusion, no pressure)
        let mut u_star = self.u.clone();
        let mut v_star = self.v.clone();

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let cell = idx(i, j, ny);
                let ui = self.u[cell];
                let vi = self.v[cell];

                // Advection: -(u·∇)u via upwind
                let dudx = if ui > 0.0 {
                    (self.u[cell] - self.u[idx(i - 1, j, ny)]) / dx
                } else {
                    (self.u[idx(i + 1, j, ny)] - self.u[cell]) / dx
                };
                let dudy = if vi > 0.0 {
                    (self.u[cell] - self.u[idx(i, j - 1, ny)]) / dx
                } else {
                    (self.u[idx(i, j + 1, ny)] - self.u[cell]) / dx
                };

                let dvdx = if ui > 0.0 {
                    (self.v[cell] - self.v[idx(i - 1, j, ny)]) / dx
                } else {
                    (self.v[idx(i + 1, j, ny)] - self.v[cell]) / dx
                };
                let dvdy = if vi > 0.0 {
                    (self.v[cell] - self.v[idx(i, j - 1, ny)]) / dx
                } else {
                    (self.v[idx(i, j + 1, ny)] - self.v[cell]) / dx
                };

                // Diffusion: ν∇²u
                let laplacian_u = (self.u[idx(i + 1, j, ny)]
                    + self.u[idx(i - 1, j, ny)]
                    + self.u[idx(i, j + 1, ny)]
                    + self.u[idx(i, j - 1, ny)]
                    - 4.0 * self.u[cell])
                    / (dx * dx);

                let laplacian_v = (self.v[idx(i + 1, j, ny)]
                    + self.v[idx(i - 1, j, ny)]
                    + self.v[idx(i, j + 1, ny)]
                    + self.v[idx(i, j - 1, ny)]
                    - 4.0 * self.v[cell])
                    / (dx * dx);

                u_star[cell] = ui + dt * (-ui * dudx - vi * dudy + nu * laplacian_u);
                v_star[cell] = vi + dt * (-ui * dvdx - vi * dvdy + nu * laplacian_v);
            }
        }

        // Step 2: Solve pressure Poisson equation: ∇²p = ρ/dt × ∇·u*
        // Use Jacobi iteration
        let mut p_new = vec![0.0; nx * ny];
        for _ in 0..50 {
            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    let cell = idx(i, j, ny);
                    let div = (u_star[idx(i + 1, j, ny)] - u_star[idx(i - 1, j, ny)]
                        + v_star[idx(i, j + 1, ny)]
                        - v_star[idx(i, j - 1, ny)])
                        / (2.0 * dx);

                    p_new[cell] = 0.25
                        * (self.p[idx(i + 1, j, ny)]
                            + self.p[idx(i - 1, j, ny)]
                            + self.p[idx(i, j + 1, ny)]
                            + self.p[idx(i, j - 1, ny)]
                            - self.rho * dx * dx * div / dt);
                }
            }
            std::mem::swap(&mut self.p, &mut p_new);
        }

        // Step 3: Correct velocity to be divergence-free
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let cell = idx(i, j, ny);
                self.u[cell] = u_star[cell]
                    - dt / self.rho * (self.p[idx(i + 1, j, ny)] - self.p[idx(i - 1, j, ny)])
                        / (2.0 * dx);
                self.v[cell] = v_star[cell]
                    - dt / self.rho * (self.p[idx(i, j + 1, ny)] - self.p[idx(i, j - 1, ny)])
                        / (2.0 * dx);
            }
        }
    }

    /// Compute the Reynolds number Re = U*L/ν for characteristic velocity U and length L.
    pub fn reynolds_number(&self, u_char: f64, l_char: f64) -> f64 {
        u_char * l_char / self.nu
    }

    /// Compute total kinetic energy: E = ½ρ∫(u² + v²)dA
    pub fn kinetic_energy(&self) -> f64 {
        let cell_area = self.dx * self.dx;
        self.u
            .iter()
            .zip(self.v.iter())
            .map(|(u, v)| 0.5 * self.rho * (u * u + v * v) * cell_area)
            .sum()
    }

    /// Compute enstrophy (integral of vorticity²): Ω = ∫ω²dA
    pub fn enstrophy(&self) -> f64 {
        let mut total = 0.0;
        let dx = self.dx;
        let ny = self.ny;
        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                let dvdx = (self.v[idx(i + 1, j, ny)] - self.v[idx(i - 1, j, ny)]) / (2.0 * dx);
                let dudy = (self.u[idx(i, j + 1, ny)] - self.u[idx(i, j - 1, ny)]) / (2.0 * dx);
                let omega = dvdx - dudy;
                total += omega * omega * dx * dx;
            }
        }
        total
    }

    /// Set lid-driven cavity boundary conditions (top wall moves at u=U).
    pub fn set_lid_driven_cavity(&mut self, u_lid: f64) {
        let ny = self.ny;
        for i in 0..self.nx {
            // Top wall: u = U, v = 0
            self.u[idx(i, ny - 1, ny)] = u_lid;
            self.v[idx(i, ny - 1, ny)] = 0.0;
            // Bottom wall
            self.u[idx(i, 0, ny)] = 0.0;
            self.v[idx(i, 0, ny)] = 0.0;
        }
        for j in 0..self.ny {
            // Left wall
            self.u[idx(0, j, ny)] = 0.0;
            self.v[idx(0, j, ny)] = 0.0;
            // Right wall
            self.u[idx(self.nx - 1, j, ny)] = 0.0;
            self.v[idx(self.nx - 1, j, ny)] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lid_driven_cavity_develops_flow() {
        let mut sim = NavierStokes2D::new(10, 10, 0.1, 0.1); // Higher viscosity for stability

        for _ in 0..200 {
            sim.set_lid_driven_cavity(0.1); // Slower lid
            sim.step_once();
        }

        let ke = sim.kinetic_energy();
        assert!(
            ke > 0.0 && ke.is_finite(),
            "Lid-driven cavity should develop flow: KE={:.6}",
            ke
        );
    }

    #[test]
    fn test_quiescent_fluid_stays_still() {
        let mut sim = NavierStokes2D::new(20, 20, 0.05, 0.01);

        for _ in 0..10 {
            sim.step_once();
        }

        let ke = sim.kinetic_energy();
        assert!(
            ke < 1e-20,
            "Quiescent fluid should stay still: KE={:.2e}",
            ke
        );
    }

    #[test]
    fn test_enstrophy_positive_with_flow() {
        let mut sim = NavierStokes2D::new(10, 10, 0.1, 0.1);

        for _ in 0..100 {
            sim.set_lid_driven_cavity(0.1);
            sim.step_once();
        }

        let omega = sim.enstrophy();
        assert!(
            omega > 0.0 && omega.is_finite(),
            "Flowing fluid should have enstrophy: {:.6}",
            omega
        );
    }

    #[test]
    fn test_reynolds_number() {
        let sim = NavierStokes2D::new(20, 20, 0.05, 0.01);
        let re = sim.reynolds_number(1.0, 1.0);
        assert!(
            (re - 100.0).abs() < 1e-10,
            "Re = U*L/ν = 1.0*1.0/0.01 = 100"
        );
    }
}
