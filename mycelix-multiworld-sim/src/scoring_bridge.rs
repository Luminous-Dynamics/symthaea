// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Scoring Bridge — Connect ScoringModel to Simulator Governance
//!
//! Maps the pluggable [`ScoringModel`] infrastructure from bridge-common
//! into the simulator's `voting_weight` function, enabling empirical
//! comparison of different governance scoring models.
//!
//! ## How It Works
//!
//! 1. Extract civic dimensions from `CivAgent` via [`CivicDimensions`]
//! 2. Compute weighted score via the model's weights
//! 3. Apply decay via [`constitutional_envelope::apply_decay`]
//! 4. Map to tier via [`constitutional_envelope::score_to_tier`]
//! 5. Convert tier to voting weight [0.0, 1.0]

use mycelix_bridge_common::constitutional_envelope::apply_decay;
use mycelix_bridge_common::scoring_model::ModelDescriptor;
use mycelix_bridge_common::sovereign_gate::CivicTier;

use crate::agent::CivAgent;
use crate::civic_dimensions::CivicDimensions;

/// A scoring-model-based governance evaluator.
///
/// Wraps a [`ModelDescriptor`] and provides a `voting_weight()` function
/// compatible with the simulator's governance comparison infrastructure.
#[derive(Debug, Clone)]
pub struct ScoringModelGovernance {
    /// The model descriptor (weights, lambda, metadata).
    pub descriptor: ModelDescriptor,
    /// Decay rate override (if different from descriptor's lambda).
    /// Uses descriptor.lambda if None.
    pub lambda_override: Option<f64>,
}

impl ScoringModelGovernance {
    /// Create from a model descriptor.
    pub fn new(descriptor: ModelDescriptor) -> Self {
        Self {
            descriptor,
            lambda_override: None,
        }
    }

    /// Create with a custom decay rate.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda_override = Some(lambda);
        self
    }

    /// The effective decay rate.
    pub fn lambda(&self) -> f64 {
        self.lambda_override.unwrap_or(self.descriptor.lambda)
    }

    /// Compute voting weight for a CivAgent.
    ///
    /// This is the core bridge function. It:
    /// 1. Derives civic dimensions from agent state
    /// 2. Selects the right dimension array for the model's arity
    /// 3. Computes weighted score
    /// 4. Applies time decay
    /// 5. Maps to tier → voting weight
    ///
    /// `elapsed_days`: days since the agent's last verified civic interaction.
    /// For simplicity in simulation, this defaults to 0 (no decay per tick)
    /// or can be set per-tick to simulate credential expiry.
    pub fn voting_weight(&self, agent: &CivAgent, current_tick: u32, elapsed_days: f64) -> f64 {
        // Must be alive and adult (>= 15 years)
        if !agent.is_alive() || agent.age_years(current_tick) < 15.0 {
            return 0.0;
        }

        // Derive civic dimensions from agent state
        let dims = CivicDimensions::from_agent(agent, current_tick);

        // Select dimension array matching model arity
        let dim_values: Vec<f64> = match self.descriptor.weights.len() {
            3 => dims.as_3d_array().to_vec(),
            4 => dims.as_4d_array().to_vec(),
            8 => dims.as_8d_array().to_vec(),
            n => {
                // For other arities, use as many 8D values as needed
                let full = dims.as_8d_array();
                full.iter().take(n).copied().collect()
            }
        };

        // Compute weighted score
        let raw_score = self.descriptor.compute_score(&dim_values);

        // Apply decay
        let decayed = apply_decay(raw_score, self.lambda(), elapsed_days);

        // Map to tier
        let tier = CivicTier::from_score(decayed);

        // Convert tier to voting weight [0.0, 1.0]
        tier_to_vote_weight(tier)
    }

    /// Compute the full scoring breakdown for analysis (not just the weight).
    pub fn evaluate_agent(
        &self,
        agent: &CivAgent,
        current_tick: u32,
        elapsed_days: f64,
    ) -> AgentScoringResult {
        let dims = CivicDimensions::from_agent(agent, current_tick);

        let dim_values: Vec<f64> = match self.descriptor.weights.len() {
            3 => dims.as_3d_array().to_vec(),
            4 => dims.as_4d_array().to_vec(),
            8 => dims.as_8d_array().to_vec(),
            n => dims.as_8d_array().iter().take(n).copied().collect(),
        };

        let raw_score = self.descriptor.compute_score(&dim_values);
        let decayed_score = apply_decay(raw_score, self.lambda(), elapsed_days);
        let tier = CivicTier::from_score(decayed_score);
        let vote_weight = if agent.is_alive() && agent.age_years(current_tick) >= 15.0 {
            tier_to_vote_weight(tier)
        } else {
            0.0
        };

        AgentScoringResult {
            agent_id: agent.id,
            model_id: self.descriptor.model_id.clone(),
            dimensions: dim_values,
            raw_score,
            decayed_score,
            tier,
            vote_weight,
        }
    }
}

/// Detailed scoring result for a single agent.
#[derive(Debug, Clone)]
pub struct AgentScoringResult {
    pub agent_id: u64,
    pub model_id: String,
    pub dimensions: Vec<f64>,
    pub raw_score: f64,
    pub decayed_score: f64,
    pub tier: mycelix_bridge_common::sovereign_gate::CivicTier,
    pub vote_weight: f64,
}

/// Map a governance tier to a voting weight in [0.0, 1.0].
///
/// Uses the same sigmoid-like progression as the existing simulator:
/// - Observer: 0.0 (no voting)
/// - Participant: 0.3
/// - Citizen: 0.6
/// - Steward: 0.85
/// - Guardian: 1.0
fn tier_to_vote_weight(tier: mycelix_bridge_common::sovereign_gate::CivicTier) -> f64 {
    use mycelix_bridge_common::sovereign_gate::CivicTier;
    match tier {
        CivicTier::Observer => 0.0,
        CivicTier::Participant => 0.3,
        CivicTier::Citizen => 0.6,
        CivicTier::Steward => 0.85,
        CivicTier::Guardian => 1.0,
    }
}

/// Create a ScoringModelGovernance for each of the 3 built-in models.
pub fn builtin_scoring_models() -> Vec<ScoringModelGovernance> {
    use mycelix_bridge_common::scoring_model::{
        Canonical4D, MinimalCivic, ScoringModel, Sovereign8D,
    };

    let models: Vec<Box<dyn ScoringModel>> = vec![
        Box::new(Canonical4D::default()),
        Box::new(Sovereign8D::governance()),
        Box::new(MinimalCivic::default()),
    ];

    models
        .iter()
        .map(|m| {
            let desc = ModelDescriptor::from_model(m.as_ref(), 0, "simulation-harness".into());
            ScoringModelGovernance::new(desc)
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, ConsciousnessState, SkillVector};
    use crate::needs::PsychologicalNeeds;
    use mycelix_bridge_common::scoring_model::{
        Canonical4D, MinimalCivic, ScoringModel, Sovereign8D,
    };
    use mycelix_bridge_common::sovereign_gate::CivicTier;

    fn sample_agent() -> CivAgent {
        CivAgent {
            id: 1,
            birth_tick: 0,
            death_tick: None,
            sex: BiologicalSex::Female,
            world_id: 0,
            health: 0.8,
            skills: SkillVector {
                engineering: 0.6,
                agriculture: 0.3,
                medicine: 0.5,
                governance: 0.7,
                science: 0.8,
                education: 0.6,
                art_culture: 0.4,
                logistics: 0.5,
            },
            education_level: 0.7,
            consciousness: ConsciousnessState {
                level: 0.6,
                meta_awareness: 0.5,
                coherence: 0.7,
                care_activation: 0.8,
                harmonic_alignment: 0.6,
                epistemic_confidence: 0.7,
            },
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: PsychologicalNeeds::new(),
            tend_balance: 50.0,
            parent_ids: None,
            faction_id: None,
            generation: 0,
            trauma_level: 0.1,
            cumulative_dose_sv: 0.01,
            adversarial: None,
            coordination_understanding: 0.5,
            mycel_score: 0.1,
            sap_balance: 100.0,
            is_biological: true,
            wounds: Vec::new(),
            ethics: crate::agent::EthicalOrientation::default(),
            sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
            justice: crate::sub_passport::RestorativeJustice::new(),
        }
    }

    fn make_4d_bridge() -> ScoringModelGovernance {
        let model = Canonical4D::default();
        let desc = ModelDescriptor::from_model(&model, 0, "test".into());
        ScoringModelGovernance::new(desc)
    }

    fn make_8d_bridge() -> ScoringModelGovernance {
        let model = Sovereign8D::governance();
        let desc = ModelDescriptor::from_model(&model, 0, "test".into());
        ScoringModelGovernance::new(desc)
    }

    fn make_3d_bridge() -> ScoringModelGovernance {
        let model = MinimalCivic::default();
        let desc = ModelDescriptor::from_model(&model, 0, "test".into());
        ScoringModelGovernance::new(desc)
    }

    #[test]
    fn voting_weight_in_valid_range() {
        let bridge = make_4d_bridge();
        let agent = sample_agent();
        let w = bridge.voting_weight(&agent, 360, 0.0); // 30 years old
        assert!(w >= 0.0 && w <= 1.0, "Vote weight {} out of range", w);
    }

    #[test]
    fn child_has_zero_voting_weight() {
        let bridge = make_4d_bridge();
        let mut agent = sample_agent();
        agent.birth_tick = 300; // born at tick 300
        let w = bridge.voting_weight(&agent, 360, 0.0); // only 5 years old
        assert_eq!(w, 0.0, "Children should have zero voting weight");
    }

    #[test]
    fn dead_agent_has_zero_voting_weight() {
        let bridge = make_4d_bridge();
        let mut agent = sample_agent();
        agent.death_tick = Some(100);
        let w = bridge.voting_weight(&agent, 360, 0.0);
        assert_eq!(w, 0.0, "Dead agents should have zero voting weight");
    }

    #[test]
    fn all_three_models_produce_valid_weights() {
        let agent = sample_agent();
        for bridge in [make_4d_bridge(), make_8d_bridge(), make_3d_bridge()] {
            let w = bridge.voting_weight(&agent, 360, 0.0);
            assert!(
                w >= 0.0 && w <= 1.0,
                "Model '{}' produced invalid weight: {}",
                bridge.descriptor.model_id,
                w
            );
        }
    }

    #[test]
    fn decay_reduces_voting_weight() {
        let bridge = make_4d_bridge();
        let agent = sample_agent();
        let w_fresh = bridge.voting_weight(&agent, 360, 0.0);
        let w_stale = bridge.voting_weight(&agent, 360, 365.0); // 1 year decay
        assert!(
            w_stale <= w_fresh,
            "Decayed weight {} should be <= fresh weight {}",
            w_stale,
            w_fresh
        );
    }

    #[test]
    fn evaluate_agent_has_correct_dimension_count() {
        let agent = sample_agent();

        let r4 = make_4d_bridge().evaluate_agent(&agent, 360, 0.0);
        assert_eq!(r4.dimensions.len(), 4);

        let r8 = make_8d_bridge().evaluate_agent(&agent, 360, 0.0);
        assert_eq!(r8.dimensions.len(), 8);

        let r3 = make_3d_bridge().evaluate_agent(&agent, 360, 0.0);
        assert_eq!(r3.dimensions.len(), 3);
    }

    #[test]
    fn evaluate_agent_score_matches_weight() {
        let bridge = make_4d_bridge();
        let agent = sample_agent();
        let result = bridge.evaluate_agent(&agent, 360, 0.0);
        let expected_weight = tier_to_vote_weight(result.tier);
        assert_eq!(result.vote_weight, expected_weight);
    }

    #[test]
    fn builtin_models_returns_three() {
        let models = builtin_scoring_models();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].descriptor.model_id, "canonical-4d-v1");
        assert_eq!(models[1].descriptor.model_id, "sovereign-8d-v1");
        assert_eq!(models[2].descriptor.model_id, "minimal-civic-v1");
    }

    #[test]
    fn lambda_override_works() {
        let bridge = make_4d_bridge().with_lambda(0.010);
        assert_eq!(bridge.lambda(), 0.010);

        let default_bridge = make_4d_bridge();
        assert_eq!(default_bridge.lambda(), 0.002); // Canonical4D default
    }

    #[test]
    fn high_consciousness_agent_gets_high_weight() {
        let bridge = make_4d_bridge();
        let mut agent = sample_agent();
        // Max out all consciousness and skills
        agent.consciousness.level = 0.95;
        agent.consciousness.epistemic_confidence = 0.95;
        agent.consciousness.care_activation = 0.95;
        agent.consciousness.harmonic_alignment = 0.95;
        agent.consciousness.coherence = 0.95;
        agent.skills = SkillVector {
            engineering: 0.9,
            agriculture: 0.9,
            medicine: 0.9,
            governance: 0.9,
            science: 0.9,
            education: 0.9,
            art_culture: 0.9,
            logistics: 0.9,
        };
        agent.education_level = 0.95;
        agent.needs.engagement = 0.95;
        agent.needs.allostatic_load = 0.05;
        agent.tend_balance = 80.0;

        let w = bridge.voting_weight(&agent, 360, 0.0);
        assert!(
            w >= 0.6,
            "High-consciousness agent should get high weight, got {}",
            w
        );
    }

    #[test]
    fn models_can_diverge_on_same_agent() {
        let agent = sample_agent();
        let w4 = make_4d_bridge().voting_weight(&agent, 360, 0.0);
        let w8 = make_8d_bridge().voting_weight(&agent, 360, 0.0);
        let w3 = make_3d_bridge().voting_weight(&agent, 360, 0.0);
        // They don't have to diverge, but it's useful to see the values
        let _ = (w4, w8, w3); // suppress unused warnings
                              // At minimum, all should be valid
        assert!(w4 >= 0.0 && w4 <= 1.0);
        assert!(w8 >= 0.0 && w8 <= 1.0);
        assert!(w3 >= 0.0 && w3 <= 1.0);
    }
}
