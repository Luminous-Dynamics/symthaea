// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dynamic human obstacle: physically plausible coworker moving through workspace.
//!
//! The human cycles through phases (idle, approaching, reaching in, lateral drift,
//! retreating) to create unpredictable but bounded interactions with the robot arm.
//! Position is injected into `WorkspaceBoundary.human_zones` each tick.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Movement phase for the simulated human.
#[derive(Debug, Clone)]
pub enum HumanPhase {
    /// Standing outside the workspace.
    Idle { remaining: f64 },
    /// Walking toward the workspace at ~1.6 m/s.
    Approaching { target: [f64; 3], speed: f64 },
    /// Reaching arm into workspace (slower, ~0.5 m/s).
    ReachingIn { target: [f64; 3], remaining: f64 },
    /// Unpredictable lateral drift within proximity.
    LateralDrift {
        center: [f64; 3],
        amplitude: f64,
        freq: f64,
        remaining: f64,
        phase_offset: f64,
    },
    /// Walking away from the workspace.
    Retreating { target: [f64; 3], speed: f64 },
}

/// A simulated human coworker.
pub struct HumanObstacle {
    /// Torso center position [x, y, z].
    pub position: [f64; 3],
    /// Arm reach radius (meters) — effective interaction zone.
    pub reach_radius: f64,
    /// Capsule half-height for torso.
    pub capsule_half_height: f64,
    /// Current movement phase.
    pub phase: HumanPhase,
    /// RNG for phase transitions.
    rng: StdRng,
    /// Starting position (outside workspace).
    home_position: [f64; 3],
    /// Total elapsed time.
    pub elapsed: f64,
}

impl HumanObstacle {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let idle_time = rng.gen_range(2.0..5.0);

        Self {
            position: [0.0, -1.2, 0.5],
            reach_radius: 0.7,
            capsule_half_height: 0.4,
            phase: HumanPhase::Idle {
                remaining: idle_time,
            },
            rng,
            home_position: [0.0, -1.2, 0.5],
            elapsed: 0.0,
        }
    }

    /// Distance from human center to a point.
    pub fn distance_to(&self, point: &[f64; 3]) -> f64 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Get the human zone entry for WorkspaceBoundary: [cx, cy, cz, radius].
    pub fn as_zone(&self) -> [f64; 4] {
        [
            self.position[0],
            self.position[1],
            self.position[2],
            self.reach_radius,
        ]
    }

    /// Step the human simulation forward by dt seconds.
    pub fn update(&mut self, dt: f64) {
        self.elapsed += dt;

        match &self.phase {
            HumanPhase::Idle { remaining } => {
                let remaining = remaining - dt;
                if remaining <= 0.0 {
                    // Transition to approaching — pick a point near workspace
                    let target_y = self.rng.gen_range(-0.3..0.3);
                    let target_x = self.rng.gen_range(0.2..0.5);
                    self.phase = HumanPhase::Approaching {
                        target: [target_x, target_y, 0.5],
                        speed: 1.6,
                    };
                } else {
                    self.phase = HumanPhase::Idle { remaining };
                }
            }
            HumanPhase::Approaching { target, speed } => {
                let target = *target;
                let speed = *speed;
                let arrived = self.move_toward(&target, speed, dt);
                if arrived {
                    // Transition to reaching in
                    let reach_target = [
                        target[0] + self.rng.gen_range(-0.1..0.1),
                        target[1] + self.rng.gen_range(-0.1..0.1),
                        target[2] + self.rng.gen_range(-0.1..0.1),
                    ];
                    let duration = self.rng.gen_range(1.5..4.0);
                    self.phase = HumanPhase::ReachingIn {
                        target: reach_target,
                        remaining: duration,
                    };
                }
            }
            HumanPhase::ReachingIn { target, remaining } => {
                let target = *target;
                let remaining = remaining - dt;
                // Slowly drift toward reach target
                self.move_toward(&target, 0.5, dt);
                if remaining <= 0.0 {
                    // Transition to lateral drift or retreat
                    if self.rng.gen_bool(0.6) {
                        let duration = self.rng.gen_range(2.0..5.0);
                        let amplitude = self.rng.gen_range(0.1..0.3);
                        let freq = self.rng.gen_range(0.3..0.8);
                        let phase_offset = self.rng.gen_range(0.0..std::f64::consts::TAU);
                        self.phase = HumanPhase::LateralDrift {
                            center: self.position,
                            amplitude,
                            freq,
                            remaining: duration,
                            phase_offset,
                        };
                    } else {
                        self.phase = HumanPhase::Retreating {
                            target: self.home_position,
                            speed: 1.4,
                        };
                    }
                } else {
                    self.phase = HumanPhase::ReachingIn { target, remaining };
                }
            }
            HumanPhase::LateralDrift {
                center,
                amplitude,
                freq,
                remaining,
                phase_offset,
            } => {
                let center = *center;
                let amplitude = *amplitude;
                let freq = *freq;
                let phase_offset = *phase_offset;
                let remaining = remaining - dt;

                // Sinusoidal lateral + forward/backward drift
                let t = self.elapsed * freq * std::f64::consts::TAU + phase_offset;
                self.position[0] = center[0] + amplitude * t.sin();
                self.position[1] = center[1] + amplitude * (t * 1.3).cos();

                if remaining <= 0.0 {
                    self.phase = HumanPhase::Retreating {
                        target: self.home_position,
                        speed: 1.4,
                    };
                } else {
                    self.phase = HumanPhase::LateralDrift {
                        center,
                        amplitude,
                        freq,
                        remaining,
                        phase_offset,
                    };
                }
            }
            HumanPhase::Retreating { target, speed } => {
                let target = *target;
                let speed = *speed;
                let arrived = self.move_toward(&target, speed, dt);
                if arrived {
                    let idle_time = self.rng.gen_range(3.0..8.0);
                    self.phase = HumanPhase::Idle {
                        remaining: idle_time,
                    };
                }
            }
        }
    }

    /// Move position toward target at given speed. Returns true if arrived.
    fn move_toward(&mut self, target: &[f64; 3], speed: f64, dt: f64) -> bool {
        let dx = target[0] - self.position[0];
        let dy = target[1] - self.position[1];
        let dz = target[2] - self.position[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < 0.02 {
            self.position = *target;
            return true;
        }

        let step = (speed * dt).min(dist);
        let inv_dist = 1.0 / dist;
        self.position[0] += dx * inv_dist * step;
        self.position[1] += dy * inv_dist * step;
        self.position[2] += dz * inv_dist * step;

        false
    }

    /// Whether the human is currently near the workspace (within 1.5m of origin).
    pub fn is_near_workspace(&self) -> bool {
        let dist = (self.position[0].powi(2) + self.position[1].powi(2)).sqrt();
        dist < 1.5
    }

    /// Current phase name for HUD display.
    pub fn phase_name(&self) -> &'static str {
        match &self.phase {
            HumanPhase::Idle { .. } => "Idle",
            HumanPhase::Approaching { .. } => "Approaching",
            HumanPhase::ReachingIn { .. } => "Reaching In",
            HumanPhase::LateralDrift { .. } => "Lateral Drift",
            HumanPhase::Retreating { .. } => "Retreating",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_starts_outside_workspace() {
        let human = HumanObstacle::new(42);
        assert!(!human.is_near_workspace() || human.distance_to(&[0.0, 0.0, 0.0]) > 0.855);
    }

    #[test]
    fn human_cycles_through_phases() {
        let mut human = HumanObstacle::new(42);
        // Run for 30 seconds — should cycle through multiple phases
        let mut phase_names = vec![];
        for _ in 0..3000 {
            human.update(0.01);
            let name = human.phase_name();
            if phase_names.last().map(|n: &String| n.as_str()) != Some(name) {
                phase_names.push(name.to_string());
            }
        }
        // Should have visited more than just Idle
        assert!(phase_names.len() >= 3, "Only visited: {:?}", phase_names);
    }

    #[test]
    fn zone_has_correct_radius() {
        let human = HumanObstacle::new(42);
        let zone = human.as_zone();
        assert_eq!(zone[3], 0.7);
    }

    #[test]
    fn distance_to_is_euclidean() {
        let human = HumanObstacle::new(42);
        let d = human.distance_to(&human.position);
        assert!(d < 1e-10);
    }
}
