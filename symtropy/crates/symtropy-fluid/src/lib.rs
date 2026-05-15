// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SPH-based fluid dynamics for Symtropy.

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

/// A single fluid particle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FluidParticle<const D: usize> {
    pub position: SVector<f64, D>,
    pub velocity: SVector<f64, D>,
    pub force: SVector<f64, D>,
    pub density: f64,
    pub pressure: f64,
    pub mass: f64,
}

/// SPH Fluid simulation parameters.
pub struct SphConfig {
    pub smoothing_radius: f64,
    pub rest_density: f64,
    pub pressure_stiffness: f64,
    pub viscosity: f64,
}

impl Default for SphConfig {
    fn default() -> Self {
        Self {
            smoothing_radius: 1.0,
            rest_density: 1000.0,
            pressure_stiffness: 2000.0,
            viscosity: 0.1,
        }
    }
}

/// A fluid simulation instance.
pub struct FluidSimulation<const D: usize> {
    pub particles: Vec<FluidParticle<D>>,
    pub config: SphConfig,
}

impl<const D: usize> FluidSimulation<D> {
    pub fn new(config: SphConfig) -> Self {
        Self {
            particles: Vec::new(),
            config,
        }
    }

    /// Step the fluid simulation.
    pub fn step(&mut self, dt: f64, gravity: &SVector<f64, D>) {
        if dt <= 0.0 || self.particles.is_empty() {
            return;
        }

        // 1. Calculate Density and Pressure
        self.calculate_density_pressure();

        // 2. Calculate Forces (Pressure + Viscosity)
        self.calculate_forces(gravity);

        // 3. Integrate
        for p in &mut self.particles {
            p.velocity += p.force * (dt / p.density);
            p.position += p.velocity * dt;
        }
    }

    fn calculate_density_pressure(&mut self) {
        let h = self.config.smoothing_radius;
        let h2 = h * h;
        // kernel normalization factor (simplified)
        let poly6 = 315.0 / (64.0 * std::f64::consts::PI * h.powi(9));

        for i in 0..self.particles.len() {
            let mut density = 0.0;
            let pi = self.particles[i].position;

            for j in 0..self.particles.len() {
                let r = pi - self.particles[j].position;
                let r2 = r.norm_squared();
                if r2 < h2 {
                    density += self.particles[j].mass * poly6 * (h2 - r2).powi(3);
                }
            }

            self.particles[i].density = density.max(self.config.rest_density);
            self.particles[i].pressure = self.config.pressure_stiffness
                * (self.particles[i].density - self.config.rest_density);
        }
    }

    fn calculate_forces(&mut self, gravity: &SVector<f64, D>) {
        let h = self.config.smoothing_radius;
        // Gradients/Laplacians for pressure/viscosity (placeholders)

        for i in 0..self.particles.len() {
            let mut f_press = SVector::<f64, D>::zeros();
            let mut f_visc = SVector::<f64, D>::zeros();

            // Simplified force calculation (O(N^2) for initial scaffolding)
            // ... (real gradients would go here)

            self.particles[i].force = f_press + f_visc + gravity * self.particles[i].density;
        }
    }
}
