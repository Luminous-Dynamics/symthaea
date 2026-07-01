// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Underwater perturbations for AUV robustness testing.

use serde::{Deserialize, Serialize};

/// AUV perturbation types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuvPerturbation {
    /// Ocean current shift (force in body frame, Newtons).
    CurrentShift { force: [f64; 3] },
    /// Buoyancy change (ballast failure, kg displacement change).
    BuoyancyChange { delta_volume: f64 },
    /// Single thruster failure (index 0-7).
    ThrusterFailure { thruster_index: usize },
    /// Entanglement (drag increase factor).
    Entanglement { drag_multiplier: f64 },
    /// Sensor noise burst (chemical sensor corruption).
    SensorNoise { noise_std: f64 },
}

/// Scheduled perturbation with start/clear steps.
#[derive(Debug, Clone)]
pub struct ScheduledPerturbation {
    pub perturbation: AuvPerturbation,
    pub start_step: usize,
    pub clear_step: Option<usize>,
}

/// Perturbation schedule builder.
#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule {
    entries: Vec<ScheduledPerturbation>,
}

impl PerturbationSchedule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        mut self,
        perturbation: AuvPerturbation,
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

    /// Strong lateral current at step 300.
    pub fn current_shift() -> Self {
        Self::new().add(
            AuvPerturbation::CurrentShift {
                force: [20.0, 10.0, 0.0],
            },
            300,
            Some(800),
        )
    }

    /// Thruster 0 failure at step 500 (permanent).
    pub fn thruster_failure() -> Self {
        Self::new().add(
            AuvPerturbation::ThrusterFailure { thruster_index: 0 },
            500,
            None,
        )
    }

    /// Entanglement at step 200.
    pub fn entanglement() -> Self {
        Self::new().add(
            AuvPerturbation::Entanglement {
                drag_multiplier: 3.0,
            },
            200,
            Some(600),
        )
    }

    pub fn active_at(&self, step: usize) -> Vec<&AuvPerturbation> {
        self.entries
            .iter()
            .filter(|e| step >= e.start_step && e.clear_step.map_or(true, |c| step < c))
            .map(|e| &e.perturbation)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_schedule() {
        let s = PerturbationSchedule::current_shift();
        assert!(s.active_at(200).is_empty());
        assert_eq!(s.active_at(500).len(), 1);
        assert!(s.active_at(900).is_empty());
    }

    #[test]
    fn test_permanent_failure() {
        let s = PerturbationSchedule::thruster_failure();
        assert!(s.active_at(400).is_empty());
        assert_eq!(s.active_at(600).len(), 1);
        assert_eq!(s.active_at(10000).len(), 1);
    }
}
