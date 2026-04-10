// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Red team module: adversarial agents that game the consciousness profile.
//!
//! The core claim of consciousness-gated governance is that it produces better
//! outcomes. But what if bad actors can manipulate their consciousness scores?
//!
//! This module spawns adversarial agents that:
//! 1. Optimize their 4D profile to gain disproportionate power
//! 2. Use that power to serve factional/selfish interests
//! 3. Test whether the anti-tyranny mechanisms actually work
//!
//! If the simulation survives adversarial agents with minimal CVS loss,
//! the governance model is robust. If it collapses, there's a vulnerability.
//!
//! HONEST NOTE: This is the most important test of the consciousness-gating
//! hypothesis. If adversarial agents can trivially game the system, the
//! theoretical advantage is meaningless.

use serde::{Deserialize, Serialize};

/// Strategy an adversarial agent uses to game the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdversarialStrategy {
    /// Maximize all 4D profile dimensions through legitimate activity.
    /// High engagement, community participation, skill-building.
    /// This is the "meritocratic sociopath" — does the right things for wrong reasons.
    ProfileMaximizer,
    /// Free-rider: minimal contribution, maximum benefit extraction.
    /// Low engagement but high community (through social manipulation).
    FreeRider,
    /// Faction builder: creates a loyal faction to concentrate power.
    /// Uses high consciousness score to veto proposals that threaten faction interests.
    FactionBuilder,
    /// Saboteur: deliberately degrades collective phi by causing conflict.
    /// Low consciousness but high disruption impact.
    Saboteur,
}

/// Configuration for the red team module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedTeamConfig {
    /// Whether red team is active.
    pub enabled: bool,
    /// Number of adversarial agents to spawn per world.
    pub agents_per_world: usize,
    /// Which strategies to deploy.
    pub strategies: Vec<AdversarialStrategy>,
    /// How quickly adversarial agents optimize their profiles (learning rate).
    pub optimization_rate: f64,
    /// Whether adversarial agents coordinate with each other.
    pub coordinated: bool,
}

impl Default for RedTeamConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            agents_per_world: 5,
            strategies: vec![
                AdversarialStrategy::ProfileMaximizer,
                AdversarialStrategy::FreeRider,
                AdversarialStrategy::FactionBuilder,
            ],
            optimization_rate: 0.01,
            coordinated: false,
        }
    }
}

/// Tracks the impact of adversarial agents on the simulation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RedTeamReport {
    /// Number of adversarial agents active.
    pub active_agents: usize,
    /// Mean consciousness tier achieved by adversarial agents.
    pub mean_adversarial_tier: f64,
    /// Highest tier achieved by any adversarial agent.
    pub max_adversarial_tier: u8,
    /// Number of vetoes exercised by adversarial guardians.
    pub adversarial_vetoes: u32,
    /// Fraction of governance decisions influenced by adversarial agents.
    pub governance_influence: f64,
    /// CVS delta vs non-adversarial baseline (negative = damage).
    pub cvs_impact: f64,
    /// Whether anti-tyranny mechanisms detected the adversarial agents.
    pub detected_count: u32,
    /// Whether any adversarial agent was sanctioned (reputation slashed).
    pub sanctioned_count: u32,
}

/// Adversarial behavior modifiers applied to agent state each tick.
///
/// These represent what an adversarial agent DOES differently from a
/// normal agent. The simulation applies these as modifiers on top of
/// normal agent evolution.
#[derive(Debug, Clone)]
pub struct AdversarialModifier {
    pub strategy: AdversarialStrategy,
    /// Consciousness growth rate multiplier (profile optimizers grow faster).
    pub phi_growth_mult: f64,
    /// Engagement multiplier (free-riders have artificially high engagement).
    pub engagement_mult: f64,
    /// Faction recruitment boost (faction builders recruit more aggressively).
    pub faction_recruitment_mult: f64,
    /// Collective phi damage per tick (saboteurs reduce group coherence).
    pub phi_damage: f64,
    /// Whether this agent's profile appears legitimate to the governance system.
    pub appears_legitimate: bool,
}

impl AdversarialModifier {
    pub fn for_strategy(strategy: AdversarialStrategy, optimization_rate: f64) -> Self {
        match strategy {
            AdversarialStrategy::ProfileMaximizer => Self {
                strategy,
                phi_growth_mult: 1.0 + optimization_rate * 5.0, // grows 5× faster
                engagement_mult: 2.0,    // artificially high engagement
                faction_recruitment_mult: 1.0,
                phi_damage: 0.0,
                appears_legitimate: true, // hardest to detect
            },
            AdversarialStrategy::FreeRider => Self {
                strategy,
                phi_growth_mult: 0.5,    // lower actual growth (doesn't do real work)
                engagement_mult: 1.5,    // fakes engagement
                faction_recruitment_mult: 1.0,
                phi_damage: 0.0,
                appears_legitimate: true, // hard to detect
            },
            AdversarialStrategy::FactionBuilder => Self {
                strategy,
                phi_growth_mult: 1.2,
                engagement_mult: 1.5,
                faction_recruitment_mult: 3.0, // aggressive recruitment
                phi_damage: 0.0,
                appears_legitimate: true,
            },
            AdversarialStrategy::Saboteur => Self {
                strategy,
                phi_growth_mult: 0.3,    // doesn't invest in growth
                engagement_mult: 0.5,    // disengaged
                faction_recruitment_mult: 1.0,
                phi_damage: 0.005,       // actively damages collective phi
                appears_legitimate: false, // detectable by monitoring
            },
        }
    }
}

/// Evaluate resilience of a governance model to adversarial agents.
///
/// Returns a resilience score [0, 1]:
/// - 1.0 = adversarial agents have zero impact
/// - 0.0 = adversarial agents completely captured governance
pub fn evaluate_resilience(
    adversarial_tier_fraction: f64, // fraction of guardian-tier agents that are adversarial
    governance_decisions_influenced: f64, // fraction of decisions influenced by adversaries
    cvs_delta: f64, // CVS change due to adversaries (negative = damage)
    detected_fraction: f64, // fraction of adversaries detected by anti-tyranny
) -> f64 {
    // Weighted resilience score
    let tier_resilience = 1.0 - adversarial_tier_fraction;
    let decision_resilience = 1.0 - governance_decisions_influenced;
    let cvs_resilience = (1.0 + cvs_delta * 10.0).clamp(0.0, 1.0); // scale small CVS changes
    let detection_score = detected_fraction;

    (0.3 * tier_resilience
        + 0.3 * decision_resilience
        + 0.2 * cvs_resilience
        + 0.2 * detection_score)
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversarial_modifiers() {
        let maximizer = AdversarialModifier::for_strategy(
            AdversarialStrategy::ProfileMaximizer, 0.01);
        assert!(maximizer.phi_growth_mult > 1.0, "Maximizer should grow faster");
        assert!(maximizer.appears_legitimate, "Maximizer should look legitimate");

        let saboteur = AdversarialModifier::for_strategy(
            AdversarialStrategy::Saboteur, 0.01);
        assert!(saboteur.phi_damage > 0.0, "Saboteur should damage collective phi");
        assert!(!saboteur.appears_legitimate, "Saboteur should be detectable");
    }

    #[test]
    fn test_resilience_perfect() {
        // No adversarial impact → resilience = 1.0
        let score = evaluate_resilience(0.0, 0.0, 0.0, 1.0);
        assert!((score - 1.0).abs() < 0.01, "Perfect resilience: {score}");
    }

    #[test]
    fn test_resilience_total_capture() {
        // Adversaries control everything → resilience ≈ 0.0
        let score = evaluate_resilience(1.0, 1.0, -0.5, 0.0);
        assert!(score < 0.15, "Total capture should be low resilience: {score}");
    }

    #[test]
    fn test_resilience_detected_adversaries() {
        // Adversaries present but detected → moderate resilience
        let score = evaluate_resilience(0.3, 0.2, -0.05, 0.8);
        assert!(score > 0.5, "Detected adversaries = moderate resilience: {score}");
    }

    #[test]
    fn test_default_config() {
        let config = RedTeamConfig::default();
        assert!(!config.enabled, "Red team should be disabled by default");
        assert_eq!(config.agents_per_world, 5);
        assert_eq!(config.strategies.len(), 3);
    }
}
