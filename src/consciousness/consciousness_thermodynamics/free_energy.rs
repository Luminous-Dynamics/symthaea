// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

use super::critical::FluctuationStats;
use super::state::{ConsciousnessPhase, PhaseTransition, ThermodynamicState};

/// Consciousness Thermodynamics Analysis Report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicsReport {
    /// Current thermodynamic state
    pub current_state: ThermodynamicState,

    /// Recent phase transitions
    pub transitions: Vec<PhaseTransition>,

    /// Fluctuation statistics
    pub fluctuations: FluctuationStats,

    /// Free energy minimization status
    pub free_energy_status: FreeEnergyStatus,

    /// Entropy production rate
    pub entropy_production_rate: f64,

    /// Equilibrium status
    pub equilibrium_status: EquilibriumStatus,

    /// Predicted next phase
    pub predicted_phase: Option<ConsciousnessPhase>,

    /// Time to next transition (if predictable)
    pub time_to_transition: Option<f64>,

    /// Overall thermodynamic health
    pub health_score: f64,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Free energy minimization status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FreeEnergyStatus {
    /// Actively minimizing (goal-directed behavior)
    Minimizing,
    /// At local minimum (stable but not optimal)
    LocalMinimum,
    /// At global minimum (optimal coherence)
    GlobalMinimum,
    /// Increasing (entropy dominated, losing coherence)
    Increasing,
    /// Fluctuating (searching for minimum)
    Searching,
}

/// Equilibrium status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EquilibriumStatus {
    /// In thermal equilibrium
    Equilibrium,
    /// Approaching equilibrium
    Equilibrating,
    /// Far from equilibrium (active, living system)
    FarFromEquilibrium,
    /// Metastable (temporary equilibrium)
    Metastable,
}

/// Statistics for thermodynamics analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThermodynamicsStats {
    /// Total states analyzed
    pub states_analyzed: u64,

    /// Phase transitions detected
    pub transitions_detected: u64,

    /// Total entropy produced
    pub total_entropy_produced: f64,

    /// Total work extracted
    pub total_work_extracted: f64,

    /// Average temperature
    pub average_temperature: f64,

    /// Time in each phase
    pub phase_durations: [f64; 6],

    /// Current stability score
    pub stability_score: f64,
}
