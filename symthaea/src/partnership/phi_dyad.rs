use crate::core::{ContinuousHV, HDC_DIMENSION};
use symthaea_core::hdc::relational_consciousness::{RelationalAssessment, RelationMode};
use crate::phi_engine::PhiEngine;

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
        let n = input
            .ai_states
            .len()
            .min(input.human_states.len())
            .min(8); // keep it small for performance

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

