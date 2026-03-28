// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mid-flight perturbations for robustness testing.

use serde::{Deserialize, Serialize};

/// Perturbation types for helicopter robustness testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}
