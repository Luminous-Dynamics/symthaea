// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mid-flight perturbations for robustness testing.

use serde::{Deserialize, Serialize};

/// Perturbation types for helicopter robustness testing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HelicopterPerturbation {
    /// Sustained crosswind force (Newtons).
    Crosswind { force_n: f64 },
    /// Main rotor degradation (RPM reduction factor 0.0–1.0).
    RotorDegradation { efficiency: f64 },
    /// Payload drop (mass reduction in kg).
    PayloadDrop { mass_kg: f64 },
    /// Engine flameout (thrust goes to zero, autorotation required).
    EngineFlameout,
    /// Tail rotor failure (yaw authority lost).
    TailRotorFailure,
}

/// Validation failure while compiling a set of active perturbations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbationError {
    NonFiniteValue,
    EfficiencyOutOfRange,
    NegativePayload,
    PayloadDropExceedsAvailable,
}

/// Canonical aggregate effects applied by a simulator for the current step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PerturbationEffects {
    pub crosswind_force: [f64; 3],
    pub main_rotor_efficiency: f64,
    pub tail_rotor_efficiency: f64,
    pub payload_mass_drop_kg: f64,
    pub engine_available: bool,
}

impl Default for PerturbationEffects {
    fn default() -> Self {
        Self {
            crosswind_force: [0.0; 3],
            main_rotor_efficiency: 1.0,
            tail_rotor_efficiency: 1.0,
            payload_mass_drop_kg: 0.0,
            engine_available: true,
        }
    }
}

impl PerturbationEffects {
    /// Compile active declarations into one non-cumulative simulator state.
    pub fn from_active(active: &[&HelicopterPerturbation]) -> Result<Self, PerturbationError> {
        let mut effects = Self::default();
        for perturbation in active {
            match **perturbation {
                HelicopterPerturbation::Crosswind { force_n } => {
                    if !force_n.is_finite() {
                        return Err(PerturbationError::NonFiniteValue);
                    }
                    effects.crosswind_force[0] += force_n;
                }
                HelicopterPerturbation::RotorDegradation { efficiency } => {
                    if !efficiency.is_finite() {
                        return Err(PerturbationError::NonFiniteValue);
                    }
                    if !(0.0..=1.0).contains(&efficiency) {
                        return Err(PerturbationError::EfficiencyOutOfRange);
                    }
                    effects.main_rotor_efficiency = effects.main_rotor_efficiency.min(efficiency);
                }
                HelicopterPerturbation::PayloadDrop { mass_kg } => {
                    if !mass_kg.is_finite() {
                        return Err(PerturbationError::NonFiniteValue);
                    }
                    if mass_kg < 0.0 {
                        return Err(PerturbationError::NegativePayload);
                    }
                    effects.payload_mass_drop_kg += mass_kg;
                }
                HelicopterPerturbation::EngineFlameout => {
                    effects.engine_available = false;
                }
                HelicopterPerturbation::TailRotorFailure => {
                    effects.tail_rotor_efficiency = 0.0;
                }
            }
        }
        Ok(effects)
    }
}

/// Scheduled perturbation: applied at `start_step`, cleared at `clear_step`.
#[derive(Debug, Clone)]
pub struct ScheduledPerturbation {
    pub perturbation: HelicopterPerturbation,
    pub start_step: usize,
    pub clear_step: Option<usize>,
}

/// Builder for perturbation schedules.
#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule {
    entries: Vec<ScheduledPerturbation>,
}

impl PerturbationSchedule {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a perturbation to the schedule.
    pub fn add(
        mut self,
        perturbation: HelicopterPerturbation,
        start: usize,
        clear: Option<usize>,
    ) -> Self {
        self.entries.push(ScheduledPerturbation {
            perturbation,
            start_step: start,
            clear_step: clear,
        });
        self
    }

    /// Crosswind at 3000N starting at step 300, clearing at step 800.
    pub fn crosswind() -> Self {
        Self::new().add(
            HelicopterPerturbation::Crosswind { force_n: 3000.0 },
            300,
            Some(800),
        )
    }

    /// Engine flameout at step 500 (permanent — tests autorotation).
    pub fn engine_flameout() -> Self {
        Self::new().add(HelicopterPerturbation::EngineFlameout, 500, None)
    }

    /// Get active perturbations at the given step.
    pub fn active_at(&self, step: usize) -> Vec<&HelicopterPerturbation> {
        self.entries
            .iter()
            .filter(|e| step >= e.start_step && e.clear_step.map_or(true, |c| step < c))
            .map(|e| &e.perturbation)
            .collect()
    }

    /// Whether any perturbation is active at this step.
    pub fn is_active(&self, step: usize) -> bool {
        !self.active_at(step).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_active_at() {
        let schedule = PerturbationSchedule::crosswind();
        assert!(schedule.active_at(200).is_empty());
        assert_eq!(schedule.active_at(500).len(), 1);
        assert!(schedule.active_at(900).is_empty());
    }

    #[test]
    fn test_permanent_perturbation() {
        let schedule = PerturbationSchedule::engine_flameout();
        assert!(schedule.active_at(400).is_empty());
        assert_eq!(schedule.active_at(600).len(), 1);
        assert_eq!(schedule.active_at(10000).len(), 1); // Never clears
    }

    #[test]
    fn effects_compile_without_cumulative_payload_application() {
        let payload = HelicopterPerturbation::PayloadDrop { mass_kg: 50.0 };
        let effects = PerturbationEffects::from_active(&[&payload]).unwrap();
        assert_eq!(effects.payload_mass_drop_kg, 50.0);
        let effects_again = PerturbationEffects::from_active(&[&payload]).unwrap();
        assert_eq!(effects_again.payload_mass_drop_kg, 50.0);
    }

    #[test]
    fn effects_combine_failures_and_reject_invalid_values() {
        let degraded = HelicopterPerturbation::RotorDegradation { efficiency: 0.6 };
        let engine = HelicopterPerturbation::EngineFlameout;
        let tail = HelicopterPerturbation::TailRotorFailure;
        let effects = PerturbationEffects::from_active(&[&degraded, &engine, &tail]).unwrap();
        assert_eq!(effects.main_rotor_efficiency, 0.6);
        assert!(!effects.engine_available);
        assert_eq!(effects.tail_rotor_efficiency, 0.0);

        let invalid = HelicopterPerturbation::RotorDegradation { efficiency: 1.1 };
        assert_eq!(
            PerturbationEffects::from_active(&[&invalid]),
            Err(PerturbationError::EfficiencyOutOfRange)
        );
    }
}
