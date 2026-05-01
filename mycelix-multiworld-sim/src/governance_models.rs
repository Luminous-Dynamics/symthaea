// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Alternative governance models for A/B comparison.
//!
//! The simulator's core claim is that consciousness-gated governance (+6.9% CVS)
//! outperforms alternatives. To test this honestly, we need MULTIPLE governance
//! models that users can swap via config.
//!
//! # Models
//!
//! | Model | How Voting Power is Assigned | Theory |
//! |-------|----------------------------|--------|
//! | ConsciousnessGated | 4D trust profile (identity, reputation, community, engagement) | Mycelix (novel) |
//! | EqualWeight | Every agent votes equally (1 person = 1 vote) | Democracy |
//! | Sortition | Random selection of decision-makers each period | Athenian democracy, van Reybrouck (2016) |
//! | Meritocratic | Skill-weighted (strongest sector skill) | Technocracy, Plato's Republic |
//! | ElderCouncil | Age-weighted (elders have more authority) | Traditional societies |
//!
//! HONEST NOTE: The ConsciousnessGated model was designed by the same team that
//! built the simulator. Any comparison should be interpreted with this bias in mind.

use serde::{Deserialize, Serialize};

/// Available governance models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GovernanceModel {
    /// Consciousness-gated: voting power scales with 4D trust profile.
    /// This is the default and the model the project advocates for.
    /// BIAS WARNING: Designed by same team that built the simulator.
    ConsciousnessGated,
    /// Equal weight: every living adult votes equally. 1 person = 1 vote.
    /// The standard democratic baseline.
    EqualWeight,
    /// Sortition: randomly selected council of N agents makes decisions.
    /// CITATION: van Reybrouck (2016) "Against Elections".
    Sortition,
    /// Meritocratic: voting power proportional to highest skill level.
    /// CITATION: Inspired by Plato's philosopher-kings, modern technocracy debates.
    /// LIMITATION: Assumes skills are objectively measurable and relevant to governance.
    Meritocratic,
    /// Elder council: voting power increases with age (wisdom of experience).
    /// CITATION: Traditional governance in many Indigenous societies.
    /// LIMITATION: Assumes age correlates with governance ability.
    ElderCouncil,
}

impl Default for GovernanceModel {
    fn default() -> Self {
        Self::ConsciousnessGated
    }
}

impl GovernanceModel {
    /// Compute voting weight for an agent under this model.
    ///
    /// Returns a weight in [0, 1] that scales the agent's voting influence.
    pub fn voting_weight(
        &self,
        phi: f64,            // consciousness level [0, 1]
        age_years: f64,      // agent age
        max_skill: f64,      // highest skill level [0, 1]
        _rng_selected: bool, // whether agent was selected by sortition this period
    ) -> f64 {
        match self {
            Self::ConsciousnessGated => {
                // Sigmoid: smooth transition, no hard cutoff
                // weight = 1 / (1 + exp(-10 * (phi - 0.3)))
                1.0 / (1.0 + (-10.0 * (phi - 0.3)).exp())
            }
            Self::EqualWeight => {
                // Everyone above minimum age votes equally
                if age_years >= 15.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Sortition => {
                // Selected agents have full weight, others have none
                // Selection happens externally; here we just gate by selection flag
                if _rng_selected && age_years >= 15.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Meritocratic => {
                // Weight proportional to highest skill
                if age_years >= 15.0 {
                    max_skill.clamp(0.1, 1.0)
                } else {
                    0.0
                }
            }
            Self::ElderCouncil => {
                // Weight increases with age: peaks at 60-70, slight decline after 80
                if age_years < 15.0 {
                    return 0.0;
                }
                let age_weight = if age_years < 30.0 {
                    0.3
                } else if age_years < 50.0 {
                    0.6
                } else if age_years < 70.0 {
                    1.0
                } else {
                    0.8 // slight decline for very old
                };
                age_weight
            }
        }
    }

    /// All available models for comparison.
    pub fn all() -> &'static [GovernanceModel] {
        &[
            Self::ConsciousnessGated,
            Self::EqualWeight,
            Self::Sortition,
            Self::Meritocratic,
            Self::ElderCouncil,
        ]
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ConsciousnessGated => "Consciousness-Gated",
            Self::EqualWeight => "Equal Weight (Democracy)",
            Self::Sortition => "Sortition (Random Council)",
            Self::Meritocratic => "Meritocratic (Skill-Weighted)",
            Self::ElderCouncil => "Elder Council (Age-Weighted)",
        }
    }

    /// Parse from string (for TOML config).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "consciousness" | "consciousnessgated" | "consciousness_gated" => {
                Some(Self::ConsciousnessGated)
            }
            "equal" | "equalweight" | "equal_weight" | "democracy" => Some(Self::EqualWeight),
            "sortition" | "random" | "random_council" => Some(Self::Sortition),
            "meritocratic" | "merit" | "skill" | "technocracy" => Some(Self::Meritocratic),
            "elder" | "eldercouncil" | "elder_council" | "seniority" => Some(Self::ElderCouncil),
            _ => None,
        }
    }
}

/// Null model for testing: assigns RANDOM weights.
/// If consciousness-gated performs no better than random, the effect is structural.
pub fn null_model_weight(rng_value: f64) -> f64 {
    rng_value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_gated_sigmoid() {
        let model = GovernanceModel::ConsciousnessGated;
        // Low phi → low weight
        assert!(model.voting_weight(0.1, 30.0, 0.5, false) < 0.2);
        // High phi → high weight
        assert!(model.voting_weight(0.8, 30.0, 0.5, false) > 0.9);
        // Transition around phi=0.3
        let mid = model.voting_weight(0.3, 30.0, 0.5, false);
        assert!((mid - 0.5).abs() < 0.1, "Midpoint should be ~0.5: {mid}");
    }

    #[test]
    fn test_equal_weight_uniform() {
        let model = GovernanceModel::EqualWeight;
        // All adults equal
        assert_eq!(model.voting_weight(0.1, 30.0, 0.2, false), 1.0);
        assert_eq!(model.voting_weight(0.9, 30.0, 0.9, false), 1.0);
        // Children excluded
        assert_eq!(model.voting_weight(0.5, 10.0, 0.5, false), 0.0);
    }

    #[test]
    fn test_meritocratic_skill_scaling() {
        let model = GovernanceModel::Meritocratic;
        let low_skill = model.voting_weight(0.5, 30.0, 0.2, false);
        let high_skill = model.voting_weight(0.5, 30.0, 0.9, false);
        assert!(
            high_skill > low_skill,
            "Higher skill = more weight: {high_skill} vs {low_skill}"
        );
    }

    #[test]
    fn test_elder_council_age_progression() {
        let model = GovernanceModel::ElderCouncil;
        let young = model.voting_weight(0.5, 25.0, 0.5, false);
        let middle = model.voting_weight(0.5, 45.0, 0.5, false);
        let elder = model.voting_weight(0.5, 65.0, 0.5, false);
        assert!(
            elder > middle,
            "Elders should have more weight: {elder} vs {middle}"
        );
        assert!(middle > young, "Middle age > young: {middle} vs {young}");
    }

    #[test]
    fn test_all_models_listed() {
        assert_eq!(GovernanceModel::all().len(), 5);
    }

    #[test]
    fn test_parse_governance_model() {
        assert_eq!(
            GovernanceModel::from_str("consciousness"),
            Some(GovernanceModel::ConsciousnessGated)
        );
        assert_eq!(
            GovernanceModel::from_str("democracy"),
            Some(GovernanceModel::EqualWeight)
        );
        assert_eq!(
            GovernanceModel::from_str("sortition"),
            Some(GovernanceModel::Sortition)
        );
        assert_eq!(
            GovernanceModel::from_str("technocracy"),
            Some(GovernanceModel::Meritocratic)
        );
        assert_eq!(
            GovernanceModel::from_str("elder"),
            Some(GovernanceModel::ElderCouncil)
        );
        assert_eq!(GovernanceModel::from_str("invalid"), None);
    }

    #[test]
    fn test_null_model_is_random() {
        // Null model should just pass through the random value
        assert!((null_model_weight(0.5) - 0.5).abs() < 0.001);
        assert!((null_model_weight(0.0) - 0.0).abs() < 0.001);
        assert!((null_model_weight(1.0) - 1.0).abs() < 0.001);
    }
}
