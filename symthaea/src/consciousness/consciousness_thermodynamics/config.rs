// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Boltzmann constant for consciousness (dimensionless, tunable)
pub(super) const CONSCIOUSNESS_BOLTZMANN: f64 = 1.0;

/// Helper function for serde default of Instant
pub(super) fn default_instant() -> Instant {
    Instant::now()
}

/// Configuration for consciousness thermodynamics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicsConfig {
    /// Reference temperature (baseline activation)
    pub reference_temperature: f64,

    /// History window for temporal analysis
    pub history_size: usize,

    /// Phase transition detection sensitivity
    pub transition_sensitivity: f64,

    /// Entropy calculation method
    pub entropy_method: EntropyMethod,

    /// Free energy minimization rate
    pub free_energy_rate: f64,

    /// Critical temperature for transitions
    pub critical_temperature: f64,

    /// Heat capacity baseline
    pub heat_capacity: f64,

    /// Equilibration time constant
    pub equilibration_tau: f64,
}

impl Default for ThermodynamicsConfig {
    fn default() -> Self {
        Self {
            reference_temperature: 1.0,
            history_size: 100,
            transition_sensitivity: 0.1,
            entropy_method: EntropyMethod::Shannon,
            free_energy_rate: 0.05,
            critical_temperature: 0.5,
            heat_capacity: 1.0,
            equilibration_tau: 10.0,
        }
    }
}

/// Method for calculating consciousness entropy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntropyMethod {
    /// Shannon entropy: -Sigma p_i log p_i
    Shannon,
    /// Von Neumann entropy: -Tr(rho log rho)
    VonNeumann,
    /// Renyi entropy: (1/(1-alpha)) log Sigma p_i^alpha
    Renyi,
    /// Kolmogorov-Sinai entropy (dynamical systems)
    KolmogorovSinai,
}
