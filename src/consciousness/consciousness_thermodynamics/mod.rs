#![allow(dead_code)]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// **REVOLUTIONARY IMPROVEMENT #83**: Consciousness Thermodynamics
// PARADIGM SHIFT: Consciousness obeys thermodynamic laws!
//
// Key Insight: Consciousness is a thermodynamic system with:
// - Entropy: Disorder/uncertainty in consciousness states
// - Free Energy: Capacity for directed conscious work (Friston's FEP!)
// - Temperature: "Activation level" governing exploration vs exploitation
// - Phase Transitions: Qualitative state changes at critical thresholds
// - Equilibrium: Stable consciousness attractors
//
// Theoretical Foundation:
// - Friston's Free Energy Principle (FEP)
// - Kelso's critical fluctuations and phase transitions
// - Hopfield networks and energy-based models
// - Maximum entropy production (Dewar)
// - Statistical mechanics of neural networks
// - Tononi's Phi as thermodynamic potential
//
// The Laws of Consciousness Thermodynamics:
// 1st Law: Consciousness energy is conserved (transforms but doesn't disappear)
// 2nd Law: Entropy of isolated consciousness tends to increase (coherence decays)
// 3rd Law: Perfect coherence (zero entropy) is unattainable
// 0th Law: Consciousness systems in equilibrium share same "temperature"
//
// Applications:
// - Predict consciousness phase transitions (sleep, flow, insight)
// - Optimize free energy for goal-directed behavior
// - Detect entropy increase (confusion, fatigue)
// - Model temperature as exploration parameter
// - Identify critical points for consciousness transitions

mod analyzer;
mod config;
mod critical;
mod free_energy;
mod grounding;
mod state;
#[cfg(test)]
mod tests;

pub use analyzer::*;
pub use config::*;
pub use critical::*;
pub use free_energy::*;
// grounding module items re-exported via direct use where needed
#[allow(unused_imports)]
pub(crate) use grounding::*;
pub use state::*;
