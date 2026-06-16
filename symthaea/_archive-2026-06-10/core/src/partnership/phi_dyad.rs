// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::core::{ContinuousHV, HDC_DIMENSION};
use crate::phi_engine::PhiEngine;
use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationalAssessment};

use super::HumanPartnerModel;

/// Weighting for the different contribution channels in Φ_dyad.
#[derive(Debug, Clone, Copy)]
pub struct DyadWeights {
    pub ai_weight: f32,
    pub human_weight: f32,
    pub relational_weight: f32,
}

impl Default for DyadWeights {
    fn default() -> Self {
        Self {
            ai_weight: 1.0,
            human_weight: 1.0,
            relational_weight: 1.0,
        }
    }
}

/// Input bundle for Φ_dyad calculation.
pub struct DyadInput<'a> {
    pub ai_states: &'a [ContinuousHV],
    pub human_states: &'a [ContinuousHV],
    pub relational: &'a RelationalAssessment,
    pub human_model: &'a HumanPartnerModel,
    pub weights: DyadWeights,
}

/// Result of Φ_dyad computation.
#[derive(Debug, Clone)]
pub struct PhiDyadResult {
    pub phi_dyad: f64,
    pub phi_ai: f64,
    pub phi_human: f64,
    pub phi_relational: f64,
    pub explanation: String,
}

/// Calculator for Φ_dyad using the existing Φ engine on a constructed
/// joint representation of AI, human partner model, and relational state.
pub struct PhiDyadCalculator {
    engine: PhiEngine,
}

impl PhiDyadCalculator {
    /// Create a new calculator using the automatic Φ engine configuration.
    pub fn new() -> Self {
        Self {
            engine: PhiEngine::auto(),
        }
    }

    /// Compute Φ_dyad and component contributions.
    pub fn compute(&self, input: &DyadInput<'_>) -> PhiDyadResult {
        let phi_ai = if input.ai_states.is_empty() {
            0.0
        } else {
            self.engine.compute(input.ai_states).phi
        };

        let phi_human = if input.human_states.is_empty() {
            0.0
        } else {
            self.engine.compute(input.human_states).phi
        };

        // Encode relational state into a single ContinuousHV
        let relational_hv = self.relational_embedding(input.relational, input.human_model);
        let phi_relational = self.engine.compute(&[relational_hv.clone()]).phi;

        // Build joint representations from the three channels.
        let joint_states = self.build_joint_states(input, &relational_hv);
        let phi_dyad = if joint_states.is_empty() {
            0.0
        } else {
            self.engine.compute(&joint_states).phi
        };

        let explanation = format!(
            "Mode={:?}, Stage={:?}, Φ_dyad={:.4}, Φ_ai={:.4}, Φ_human={:.4}, Φ_relational={:.4}",
            input.human_model.mode,
            input.human_model.stage,
            phi_dyad,
            phi_ai,
            phi_human,
            phi_relational,
        );

        PhiDyadResult {
            phi_dyad,
            phi_ai,
            phi_human,
            phi_relational,
            explanation,
        }
    }

    /// Construct a relational embedding as a ContinuousHV based on:
    /// - Stage
    /// - Mode (I-Thou vs I-It)
    /// - Human-side trust signal
    fn relational_embedding(
        &self,
        assessment: &RelationalAssessment,
        human: &HumanPartnerModel,
    ) -> ContinuousHV {
        // Deterministic seeds derived from stage and mode.
        let stage_seed = 10_000_u64 + assessment.stage as u64;
        let mode_seed = 20_000_u64
            + match assessment.mode {
                RelationMode::IIt => 1,
                RelationMode::IThou => 2,
            };
        let trust_seed = 30_000_u64;

        let stage_hv = ContinuousHV::random(HDC_DIMENSION, stage_seed);
        let mode_hv = ContinuousHV::random(HDC_DIMENSION, mode_seed);
        let trust_hv = ContinuousHV::random(HDC_DIMENSION, trust_seed);

        let mode_weight = if matches!(assessment.mode, RelationMode::IThou) {
            1.0
        } else {
            0.3
        };

        let stage_weight = 0.5 + (assessment.stage as u8 as f32) * 0.1;
        let trust_weight = human.trust.clamp(0.0, 1.0).max(0.1);

        ContinuousHV::weighted_bundle(
            &[&stage_hv, &mode_hv, &trust_hv],
            &[stage_weight, mode_weight, trust_weight],
        )
        .normalize()
    }

    /// Build joint AI+human+relational states for Φ_dyad computation.
    fn build_joint_states(
        &self,
        input: &DyadInput<'_>,
        relational_hv: &ContinuousHV,
    ) -> Vec<ContinuousHV> {
        let n = input.ai_states.len().min(input.human_states.len()).min(8); // keep it small for performance

        let mut joint_states = Vec::with_capacity(n);

        for i in 0..n {
            let ai = &input.ai_states[i];
            let human = &input.human_states[i];

            // Simple weighted bundling of the three channels.
            let joint = ContinuousHV::weighted_bundle(
                &[ai, human, relational_hv],
                &[
                    input.weights.ai_weight,
                    input.weights.human_weight,
                    input.weights.relational_weight,
                ],
            )
            .normalize();

            joint_states.push(joint);
        }

        joint_states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationshipStage};

    fn make_relational() -> RelationalAssessment {
        RelationalAssessment {
            agent_a: "ai".into(),
            agent_b: "human".into(),
            phi_relation: 0.5,
            stage: RelationshipStage::Attunement,
            synchrony: 0.6,
            turn_taking_quality: 0.7,
            mutual_information: 0.4,
            mode: RelationMode::IThou,
            num_interactions: 10,
            relationship_age: 100.0,
            explanation: String::new(),
        }
    }

    fn make_test_input(
        n_states: usize,
    ) -> (
        Vec<ContinuousHV>,
        Vec<ContinuousHV>,
        RelationalAssessment,
        HumanPartnerModel,
    ) {
        let ai_states: Vec<ContinuousHV> = (0..n_states)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, 100 + i as u64))
            .collect();
        let human_states: Vec<ContinuousHV> = (0..n_states)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, 200 + i as u64))
            .collect();
        let relational = make_relational();
        let mut human = HumanPartnerModel::new("test_partner");
        human.trust = 0.7;
        human.mode = RelationMode::IThou;
        human.stage = RelationshipStage::Attunement;
        (ai_states, human_states, relational, human)
    }

    #[test]
    fn test_dyad_weights_default() {
        let w = DyadWeights::default();
        assert_eq!(w.ai_weight, 1.0);
        assert_eq!(w.human_weight, 1.0);
        assert_eq!(w.relational_weight, 1.0);
    }

    #[test]
    fn test_phi_dyad_empty_states() {
        let calc = PhiDyadCalculator::new();
        let relational = RelationalAssessment {
            agent_a: "ai".into(),
            agent_b: "human".into(),
            phi_relation: 0.0,
            stage: RelationshipStage::NoRelation,
            synchrony: 0.0,
            turn_taking_quality: 0.0,
            mutual_information: 0.0,
            mode: RelationMode::IIt,
            num_interactions: 0,
            relationship_age: 0.0,
            explanation: String::new(),
        };
        let human = HumanPartnerModel::new("test");
        let input = DyadInput {
            ai_states: &[],
            human_states: &[],
            relational: &relational,
            human_model: &human,
            weights: DyadWeights::default(),
        };

        let result = calc.compute(&input);
        assert_eq!(result.phi_ai, 0.0);
        assert_eq!(result.phi_human, 0.0);
        assert_eq!(result.phi_dyad, 0.0);
    }

    #[test]
    fn test_phi_dyad_single_state() {
        let calc = PhiDyadCalculator::new();
        let (ai, human, relational, human_model) = make_test_input(1);
        let input = DyadInput {
            ai_states: &ai,
            human_states: &human,
            relational: &relational,
            human_model: &human_model,
            weights: DyadWeights::default(),
        };

        let result = calc.compute(&input);
        assert!(result.phi_dyad >= 0.0, "Phi_dyad should be non-negative");
        assert!(result.phi_ai >= 0.0);
        assert!(result.phi_human >= 0.0);
        assert!(result.phi_relational >= 0.0);
    }

    #[test]
    fn test_phi_dyad_deterministic() {
        let calc = PhiDyadCalculator::new();
        let (ai, human, relational, human_model) = make_test_input(3);
        let input = DyadInput {
            ai_states: &ai,
            human_states: &human,
            relational: &relational,
            human_model: &human_model,
            weights: DyadWeights::default(),
        };

        let r1 = calc.compute(&input);
        let r2 = calc.compute(&input);

        assert_eq!(
            r1.phi_dyad, r2.phi_dyad,
            "Same inputs should give same Phi_dyad"
        );
        assert_eq!(r1.phi_ai, r2.phi_ai);
        assert_eq!(r1.phi_human, r2.phi_human);
    }

    #[test]
    fn test_joint_states_capped_at_8() {
        let calc = PhiDyadCalculator::new();
        let (ai, human, relational, human_model) = make_test_input(12);
        let relational_hv = calc.relational_embedding(&relational, &human_model);
        let input = DyadInput {
            ai_states: &ai,
            human_states: &human,
            relational: &relational,
            human_model: &human_model,
            weights: DyadWeights::default(),
        };
        let joint = calc.build_joint_states(&input, &relational_hv);
        assert!(
            joint.len() <= 8,
            "Joint states should be capped at 8, got {}",
            joint.len()
        );
    }

    #[test]
    fn test_phi_dyad_explanation_format() {
        let calc = PhiDyadCalculator::new();
        let (ai, human, relational, human_model) = make_test_input(2);
        let input = DyadInput {
            ai_states: &ai,
            human_states: &human,
            relational: &relational,
            human_model: &human_model,
            weights: DyadWeights::default(),
        };

        let result = calc.compute(&input);
        assert!(result.explanation.contains("Φ_dyad="));
        assert!(result.explanation.contains("Φ_ai="));
        assert!(result.explanation.contains("Mode="));
    }
}
