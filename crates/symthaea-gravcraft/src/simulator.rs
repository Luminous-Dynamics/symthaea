// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metric perturbation simulator — geodesic equation, NOT F=ma.
//!
//! The gravcraft moves through spacetime by following geodesics in a
//! perturbed metric. Position updates via:
//!
//! dx^i/dt = v^i
//! dv^i/dt = -Γ^i_jk v^j v^k + β^i_shift
//!
//! where Γ are Christoffel symbols computed from the metric perturbation,
//! and β is the shift vector from the gravity amplifiers.

use crate::amplifier::combined_metric;
use crate::types::{GravcraftConfig, GravcraftState, MetricCommand};

/// Metric perturbation simulator.
///
/// This is fundamentally different from all other Symthaea simulators:
/// the craft doesn't experience forces — spacetime itself is modified.
pub struct MetricPerturbationSimulator {
    state: GravcraftState,
    config: GravcraftConfig,
}

impl MetricPerturbationSimulator {
    pub fn new(config: GravcraftConfig) -> Self {
        Self {
            state: GravcraftState::default(),
            config,
        }
    }

    pub fn state(&self) -> &GravcraftState {
        &self.state
    }

    pub fn reset(&mut self) {
        self.state = GravcraftState::default();
    }

    /// Advance the simulation by dt seconds using the given metric command.
    ///
    /// Uses RK4 integration of the geodesic equation.
    pub fn step(&mut self, cmd: &MetricCommand, dt: f64) {
        let substeps = self.config.rk4_substeps.max(1);
        let sub_dt = dt / substeps as f64;

        for _ in 0..substeps {
            self.rk4_step(cmd, sub_dt);
        }

        self.state.timestamp += dt;

        // Update metric diagnostics
        let (_, mag) = combined_metric(
            self.state.position,
            self.state.position,
            &cmd.amplifiers,
            &self.config,
        );
        self.state.metric_perturbation = mag;

        // Tidal tensor norm ~ gradient of metric shift (finite difference)
        self.state.tidal_tensor_norm = self.estimate_tidal_norm(cmd);

        // Record amplifier states
        for (i, &(amp, focal, az, el)) in cmd.amplifiers.iter().enumerate() {
            self.state.amplifiers[i].amplitude = amp;
            self.state.amplifiers[i].focal_distance = focal;
            self.state.amplifiers[i].azimuth = az;
            self.state.amplifiers[i].elevation = el;
        }
    }

    /// Single RK4 substep for the coupled position-velocity ODE system.
    fn rk4_step(&mut self, cmd: &MetricCommand, dt: f64) {
        let pos = self.state.position;
        let vel = self.state.velocity;

        // k1: derivatives at (pos, vel)
        let (a1, _) = self.acceleration(pos, vel, cmd);
        let v1 = vel;

        // k2: derivatives at (pos + 0.5*dt*v1, vel + 0.5*dt*a1)
        let p2 = [
            pos[0] + 0.5 * dt * v1[0],
            pos[1] + 0.5 * dt * v1[1],
            pos[2] + 0.5 * dt * v1[2],
        ];
        let u2 = [
            vel[0] + 0.5 * dt * a1[0],
            vel[1] + 0.5 * dt * a1[1],
            vel[2] + 0.5 * dt * a1[2],
        ];
        let (a2, _) = self.acceleration(p2, u2, cmd);

        // k3: derivatives at (pos + 0.5*dt*v2, vel + 0.5*dt*a2)
        let p3 = [
            pos[0] + 0.5 * dt * u2[0],
            pos[1] + 0.5 * dt * u2[1],
            pos[2] + 0.5 * dt * u2[2],
        ];
        let u3 = [
            vel[0] + 0.5 * dt * a2[0],
            vel[1] + 0.5 * dt * a2[1],
            vel[2] + 0.5 * dt * a2[2],
        ];
        let (a3, _) = self.acceleration(p3, u3, cmd);

        // k4: derivatives at (pos + dt*v3, vel + dt*a3)
        let p4 = [
            pos[0] + dt * u3[0],
            pos[1] + dt * u3[1],
            pos[2] + dt * u3[2],
        ];
        let u4 = [
            vel[0] + dt * a3[0],
            vel[1] + dt * a3[1],
            vel[2] + dt * a3[2],
        ];
        let (a4, _) = self.acceleration(p4, u4, cmd);

        // RK4 combination
        for i in 0..3 {
            self.state.position[i] =
                pos[i] + dt / 6.0 * (v1[i] + 2.0 * u2[i] + 2.0 * u3[i] + u4[i]);
            self.state.velocity[i] =
                vel[i] + dt / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
        }
    }

    /// Compute acceleration from the geodesic equation + metric shift.
    ///
    /// dv^i/dt = β^i_shift (simplified: shift vector acts as coordinate acceleration)
    ///
    /// In full GR: dv^i/dt = -Γ^i_jk v^j v^k, but at low velocities and
    /// weak perturbations, the dominant effect is the shift vector gradient.
    fn acceleration(
        &self,
        position: [f64; 3],
        _velocity: [f64; 3],
        cmd: &MetricCommand,
    ) -> ([f64; 3], [f64; 3]) {
        let (shift, _) = combined_metric(
            position,
            self.state.position, // amplifiers centered on craft
            &cmd.amplifiers,
            &self.config,
        );

        // The shift vector gradient provides the "force" (coordinate acceleration)
        // In the weak-field limit: a^i ≈ ∂β^i/∂t ≈ dβ^i/dx^j × v^j + β^i
        // For simplicity we use the shift vector directly as acceleration
        let accel = shift;

        // Velocity is just position derivative
        let vel = [
            _velocity[0] + shift[0],
            _velocity[1] + shift[1],
            _velocity[2] + shift[2],
        ];

        (accel, vel)
    }

    /// Estimate tidal tensor norm from metric gradients (finite difference).
    fn estimate_tidal_norm(&self, cmd: &MetricCommand) -> f64 {
        let eps = 0.1; // meters
        let pos = self.state.position;
        let mut grad_sum = 0.0;

        for axis in 0..3 {
            let mut pos_plus = pos;
            let mut pos_minus = pos;
            pos_plus[axis] += eps;
            pos_minus[axis] -= eps;

            let (shift_plus, _) = combined_metric(pos_plus, pos, &cmd.amplifiers, &self.config);
            let (shift_minus, _) = combined_metric(pos_minus, pos, &cmd.amplifiers, &self.config);

            for i in 0..3 {
                let grad = (shift_plus[i] - shift_minus[i]) / (2.0 * eps);
                grad_sum += grad * grad;
            }
        }

        grad_sum.sqrt()
    }
}

impl Default for MetricPerturbationSimulator {
    fn default() -> Self {
        Self::new(GravcraftConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_metric_no_motion() {
        let mut sim = MetricPerturbationSimulator::default();
        let cmd = MetricCommand::default(); // All zeros → flat metric
        sim.step(&cmd, 1.0);

        // With flat metric, craft should not move
        let pos = sim.state().position;
        assert!(
            pos[0].abs() < 1e-10 && pos[1].abs() < 1e-10 && pos[2].abs() < 1e-10,
            "Flat metric should produce no motion: {:?}",
            pos
        );
    }

    #[test]
    fn test_metric_perturbation_produces_motion() {
        let mut sim = MetricPerturbationSimulator::default();
        // Single amplifier pointing forward (+x)
        let cmd = MetricCommand {
            amplifiers: [
                (0.05, 10.0, 0.0, 0.0), // Forward
                (0.0, 0.0, 0.0, 0.0),   // Off
                (0.0, 0.0, 0.0, 0.0),   // Off
            ],
        };

        // Run multiple steps
        for _ in 0..100 {
            sim.step(&cmd, 0.01);
        }

        // Craft should have moved
        let pos = sim.state().position;
        let distance = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
        assert!(
            distance > 1e-15,
            "Metric perturbation should produce non-zero motion: distance={}",
            distance
        );
    }

    #[test]
    fn test_simulator_deterministic() {
        let mut sim1 = MetricPerturbationSimulator::default();
        let mut sim2 = MetricPerturbationSimulator::default();
        let cmd = MetricCommand {
            amplifiers: [
                (0.03, 8.0, 0.5, 0.1),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ],
        };

        for _ in 0..10 {
            sim1.step(&cmd, 0.01);
            sim2.step(&cmd, 0.01);
        }

        assert_eq!(sim1.state().position, sim2.state().position);
    }

    #[test]
    fn test_metric_perturbation_reported() {
        let mut sim = MetricPerturbationSimulator::default();
        let cmd = MetricCommand {
            amplifiers: [
                (0.05, 10.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ],
        };
        sim.step(&cmd, 0.01);
        assert!(
            sim.state().metric_perturbation > 0.0,
            "Metric perturbation should be reported: {}",
            sim.state().metric_perturbation
        );
    }

    #[test]
    fn test_emergency_neutralization() {
        let mut sim = MetricPerturbationSimulator::default();
        // First apply metric perturbation
        let cmd = MetricCommand {
            amplifiers: [
                (0.05, 10.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ],
        };
        sim.step(&cmd, 0.1);

        // Then neutralize (all zeros)
        let flat_cmd = MetricCommand::default();
        sim.step(&flat_cmd, 0.01);

        assert!(
            sim.state().metric_perturbation < 1e-10,
            "Emergency neutralization should produce flat metric: {}",
            sim.state().metric_perturbation
        );
    }
}
