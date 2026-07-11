// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Configuration for the Master Consciousness Equation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterEquationConfig {
    /// Temperature for softmin (lower = more bottleneck-sensitive)
    pub softmin_tau: f64,

    /// Minimum value to prevent division by zero
    pub epsilon: f64,

    /// Weights for each component
    pub component_weights: ComponentWeights,

    /// Enable/disable new factors
    pub enable_embodiment_factor: bool,
    pub enable_narrative_factor: bool,
    pub enable_social_factor: bool,

    /// Temporal stability window (ms)
    pub temporal_window_ms: u64,

    /// History size for temporal averaging
    pub history_size: usize,
}

impl Default for MasterEquationConfig {
    fn default() -> Self {
        Self {
            // τ=0.25: conservative step toward evolution-found optimal of 0.35.
            // Softer bottleneck allows partial compensation from strong factors
            // when one factor is weak — matching neuroscience: consciousness degrades
            // gracefully, not catastrophically (Baars 2005; Dehaene 2014).
            // τ=0.15 was too bottleneck-sensitive for text cognition (cold_start
            // failed to reach C>0.50). τ=0.35 risks over-inflating; 0.25 balances.
            softmin_tau: 0.25,
            epsilon: 1e-8,
            component_weights: ComponentWeights::default(),
            enable_embodiment_factor: true,
            enable_narrative_factor: true,
            enable_social_factor: true,
            temporal_window_ms: 1000,
            history_size: 100,
        }
    }
}

/// Weights for consciousness components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentWeights {
    /// IIT Phi weight
    pub phi: f64,
    /// GWT Broadcast weight
    pub broadcast: f64,
    /// Working memory weight
    pub working_memory: f64,
    /// Attention weight
    pub attention: f64,
    /// Recurrence weight
    pub recurrence: f64,
    /// Embodiment weight
    pub embodiment: f64,
    /// Knowledge weight
    pub knowledge: f64,
    /// New: Embodiment factor weight
    pub embodiment_factor: f64,
    /// New: Narrative coherence weight
    pub narrative: f64,
    /// New: Social embedding weight
    pub social: f64,
}

impl Default for ComponentWeights {
    fn default() -> Self {
        Self {
            phi: 0.15,
            broadcast: 0.10,
            working_memory: 0.10,
            attention: 0.12,
            recurrence: 0.10,
            embodiment: 0.10,
            knowledge: 0.08,
            embodiment_factor: 0.10,
            narrative: 0.08,
            social: 0.07,
        }
    }
}

impl ComponentWeights {
    /// Get total weight sum for normalization
    pub fn total(&self) -> f64 {
        self.phi
            + self.broadcast
            + self.working_memory
            + self.attention
            + self.recurrence
            + self.embodiment
            + self.knowledge
            + self.embodiment_factor
            + self.narrative
            + self.social
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.total();
        if total > 0.0 {
            self.phi /= total;
            self.broadcast /= total;
            self.working_memory /= total;
            self.attention /= total;
            self.recurrence /= total;
            self.embodiment /= total;
            self.knowledge /= total;
            self.embodiment_factor /= total;
            self.narrative /= total;
            self.social /= total;
        }
    }
}
