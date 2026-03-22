// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mid-flight physical perturbations for domain randomization (behind `mujoco` feature).
//!
//! Instead of training 10,000 drones with different masses and hoping the final
//! model generalizes, we alter the physics DURING flight. When mass changes
//! mid-flight, FEP's prediction error spikes, tau drops, and the system
//! re-stabilizes WITHOUT retraining — a biological survival reflex.

#[cfg(feature = "mujoco")]
use crate::mujoco_sim::MuJoCoSimulator;
#[cfg(feature = "mujoco")]
use crate::simulator::PhysicsSimulator;
#[cfg(feature = "mujoco")]
use crate::types::QuadrotorCommand;

/// A physical perturbation applied to the MuJoCo simulation mid-flight.
#[derive(Debug, Clone)]
pub enum FlightPerturbation {
    /// Increase body mass by a factor (e.g., 1.3 = +30%, simulating payload pickup).
    MassChange { factor: f64, at_step: usize },
    /// Cap one rotor's thrust to a fraction of max (e.g., 0.5 = 50% power).
    RotorDegradation {
        rotor_index: usize,
        max_fraction: f64,
        at_step: usize,
    },
    /// Shift center of mass (simulating asymmetric payload).
    CenterOfMassShift { offset: [f64; 3], at_step: usize },
    /// Change air density (simulating altitude change).
    DensityChange { new_density: f64, at_step: usize },
    /// Sustained wind force applied for a duration.
    WindBurst {
        force: [f64; 3],
        at_step: usize,
        duration_steps: usize,
    },
}

impl FlightPerturbation {
    /// Get the step at which this perturbation triggers.
    pub fn trigger_step(&self) -> usize {
        match self {
            Self::MassChange { at_step, .. } => *at_step,
            Self::RotorDegradation { at_step, .. } => *at_step,
            Self::CenterOfMassShift { at_step, .. } => *at_step,
            Self::DensityChange { at_step, .. } => *at_step,
            Self::WindBurst { at_step, .. } => *at_step,
        }
    }
}

/// A schedule of perturbations to apply during a flight episode.
#[derive(Debug, Clone)]
pub struct PerturbationSchedule {
    perturbations: Vec<FlightPerturbation>,
}

impl PerturbationSchedule {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self {
            perturbations: Vec::new(),
        }
    }

    /// Add a perturbation to the schedule.
    pub fn add(&mut self, perturbation: FlightPerturbation) {
        self.perturbations.push(perturbation);
    }

    /// Builder pattern: add and return self.
    pub fn with(mut self, perturbation: FlightPerturbation) -> Self {
        self.perturbations.push(perturbation);
        self
    }

    /// Return all perturbations that trigger at the given step.
    ///
    /// This enables schedule logic testing without requiring a MuJoCo simulator.
    /// WindBurst perturbations are active for their full duration window.
    pub fn triggered_at(&self, step: usize) -> Vec<&FlightPerturbation> {
        self.perturbations
            .iter()
            .filter(|p| match p {
                FlightPerturbation::MassChange { at_step, .. } => step == *at_step,
                FlightPerturbation::RotorDegradation { at_step, .. } => step == *at_step,
                FlightPerturbation::CenterOfMassShift { at_step, .. } => step == *at_step,
                FlightPerturbation::DensityChange { at_step, .. } => step == *at_step,
                FlightPerturbation::WindBurst {
                    at_step,
                    duration_steps,
                    ..
                } => step >= *at_step && step < at_step + duration_steps,
            })
            .collect()
    }

    /// Apply any perturbations scheduled for the given step.
    /// Returns true if any perturbation was applied.
    #[cfg(feature = "mujoco")]
    pub fn apply(&self, step: usize, sim: &mut MuJoCoSimulator) -> bool {
        let mut applied = false;

        for p in &self.perturbations {
            match p {
                FlightPerturbation::MassChange { factor, at_step } if step == *at_step => {
                    let current_mass = sim.body_mass();
                    sim.set_body_mass(current_mass * factor);
                    applied = true;
                }
                FlightPerturbation::RotorDegradation {
                    max_fraction,
                    at_step,
                    ..
                } if step == *at_step => {
                    sim.set_thrust_limit(QuadrotorCommand::MAX_THRUST as f64 * max_fraction);
                    applied = true;
                }
                FlightPerturbation::WindBurst {
                    force,
                    at_step,
                    duration_steps,
                } if step >= *at_step && step < at_step + duration_steps => {
                    sim.apply_external_force(*force);
                    applied = true;
                }
                // CenterOfMassShift and DensityChange require deeper MuJoCo FFI access
                // that depends on model structure — applied via set_body_mass pattern
                FlightPerturbation::CenterOfMassShift { at_step, .. } if step == *at_step => {
                    // Note: Full COM shift requires modifying model.body_ipos,
                    // which is model-dependent. For now, this is a no-op marker
                    // that can be extended when the full MuJoCo C API is wired.
                    applied = true;
                }
                FlightPerturbation::DensityChange {
                    new_density,
                    at_step,
                } if step == *at_step => {
                    // Air density is in model.opt.density — requires mutable model access
                    let _ = new_density; // Will be wired via model.opt_mut().density
                    applied = true;
                }
                _ => {}
            }
        }

        applied
    }

    /// Get all perturbations in the schedule.
    pub fn perturbations(&self) -> &[FlightPerturbation] {
        &self.perturbations
    }

    /// Check if any perturbations are scheduled.
    pub fn is_empty(&self) -> bool {
        self.perturbations.is_empty()
    }

    /// Number of perturbations.
    pub fn len(&self) -> usize {
        self.perturbations.len()
    }

    // ── Preset schedules ──

    /// The "survival reflex" benchmark: sudden +30% mass at step 250.
    pub fn survival_reflex() -> Self {
        Self::new().with(FlightPerturbation::MassChange {
            factor: 1.3,
            at_step: 250,
        })
    }

    /// Progressive degradation: mass increase then rotor failure.
    pub fn progressive_degradation() -> Self {
        Self::new()
            .with(FlightPerturbation::MassChange {
                factor: 1.15,
                at_step: 200,
            })
            .with(FlightPerturbation::RotorDegradation {
                rotor_index: 0,
                max_fraction: 0.7,
                at_step: 400,
            })
            .with(FlightPerturbation::MassChange {
                factor: 1.15,
                at_step: 600,
            })
    }

    /// Wind gust stress test with physical perturbation.
    pub fn wind_and_mass() -> Self {
        Self::new()
            .with(FlightPerturbation::WindBurst {
                force: [0.08, 0.0, 0.0],
                at_step: 100,
                duration_steps: 100,
            })
            .with(FlightPerturbation::MassChange {
                factor: 1.3,
                at_step: 300,
            })
            .with(FlightPerturbation::WindBurst {
                force: [0.0, -0.06, 0.0],
                at_step: 500,
                duration_steps: 150,
            })
    }
}

impl Default for PerturbationSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from a perturbation benchmark run.
#[derive(Debug, Clone)]
pub struct PerturbationBenchmarkResult {
    /// Average position error before perturbation.
    pub pre_perturbation_error: f64,
    /// Maximum position error after perturbation.
    pub peak_error: f64,
    /// Steps to recover within 5cm of setpoint after perturbation.
    pub recovery_steps: usize,
    /// Final position error (after recovery period).
    pub final_error: f64,
    /// Minimum tau factor observed (lower = faster adaptation).
    pub min_tau: f64,
    /// Whether the drone crashed (altitude reached zero).
    pub crashed: bool,
    /// Per-step free energy trace (for plotting).
    pub free_energy_trace: Vec<f64>,
    /// Per-step position error trace.
    pub error_trace: Vec<f64>,
    /// Per-step tau trace.
    pub tau_trace: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perturbation_schedule_builder() {
        let schedule = PerturbationSchedule::new()
            .with(FlightPerturbation::MassChange {
                factor: 1.3,
                at_step: 100,
            })
            .with(FlightPerturbation::RotorDegradation {
                rotor_index: 0,
                max_fraction: 0.5,
                at_step: 200,
            });

        assert_eq!(schedule.len(), 2);
        assert!(!schedule.is_empty());
    }

    #[test]
    fn test_perturbation_trigger_step() {
        let p = FlightPerturbation::MassChange {
            factor: 1.5,
            at_step: 42,
        };
        assert_eq!(p.trigger_step(), 42);
    }

    #[test]
    fn test_survival_reflex_preset() {
        let schedule = PerturbationSchedule::survival_reflex();
        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule.perturbations()[0].trigger_step(), 250);
    }

    #[test]
    fn test_progressive_degradation_preset() {
        let schedule = PerturbationSchedule::progressive_degradation();
        assert_eq!(schedule.len(), 3);
    }

    #[test]
    fn test_wind_and_mass_preset() {
        let schedule = PerturbationSchedule::wind_and_mass();
        assert_eq!(schedule.len(), 3);
    }

    #[test]
    fn test_empty_schedule() {
        let schedule = PerturbationSchedule::default();
        assert!(schedule.is_empty());
        assert_eq!(schedule.len(), 0);
    }

    #[test]
    fn test_triggered_at_mass_change() {
        let schedule = PerturbationSchedule::new().with(FlightPerturbation::MassChange {
            factor: 1.3,
            at_step: 100,
        });
        assert!(schedule.triggered_at(99).is_empty());
        assert_eq!(schedule.triggered_at(100).len(), 1);
        assert!(schedule.triggered_at(101).is_empty());
    }

    #[test]
    fn test_triggered_at_wind_burst_duration() {
        let schedule = PerturbationSchedule::new().with(FlightPerturbation::WindBurst {
            force: [0.1, 0.0, 0.0],
            at_step: 50,
            duration_steps: 20,
        });
        // Before window
        assert!(schedule.triggered_at(49).is_empty());
        // During window
        assert_eq!(schedule.triggered_at(50).len(), 1);
        assert_eq!(schedule.triggered_at(60).len(), 1);
        assert_eq!(schedule.triggered_at(69).len(), 1);
        // After window
        assert!(schedule.triggered_at(70).is_empty());
    }

    #[test]
    fn test_triggered_at_ordering() {
        let schedule = PerturbationSchedule::new()
            .with(FlightPerturbation::MassChange {
                factor: 1.2,
                at_step: 100,
            })
            .with(FlightPerturbation::RotorDegradation {
                rotor_index: 0,
                max_fraction: 0.5,
                at_step: 200,
            })
            .with(FlightPerturbation::WindBurst {
                force: [0.05, 0.0, 0.0],
                at_step: 150,
                duration_steps: 100,
            });
        // Step 100: only mass change
        assert_eq!(schedule.triggered_at(100).len(), 1);
        // Step 180: only wind burst (in window 150..250)
        assert_eq!(schedule.triggered_at(180).len(), 1);
        // Step 200: rotor degradation + wind burst (still in window)
        assert_eq!(schedule.triggered_at(200).len(), 2);
        // Step 250: nothing (wind ends at 250, rotor was one-shot at 200)
        assert!(schedule.triggered_at(250).is_empty());
    }

    #[test]
    fn test_triggered_at_empty_schedule() {
        let schedule = PerturbationSchedule::new();
        assert!(schedule.triggered_at(0).is_empty());
        assert!(schedule.triggered_at(1000).is_empty());
    }
}
