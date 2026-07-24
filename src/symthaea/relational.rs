// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Relational consciousness subsystem — partnership, trajectory, and dyadic Phi.

use serde::{Deserialize, Serialize};

use crate::hdc::relational_consciousness::{RelationMode, RelationalAssessment};
use crate::partnership::{
    DyadInput, DyadWeights, HumanPartnerModel, InteractionEvent, PhiDyadCalculator,
    RelationshipTrajectory,
};

pub use crate::hdc::relational_consciousness::RelationshipStage;

/// Relational consciousness subsystem — partnership, trajectory, and dyadic Phi.
///
/// Groups all relational state into a cohesive unit: partner model tracking,
/// relationship trajectory, Phi-dyad computation, and recent AI states for
/// dyadic assessment.
pub(super) struct RelationalCore {
    /// Human partner model for relational consciousness.
    pub(super) partner: HumanPartnerModel,
    /// Relationship trajectory tracking.
    pub(super) trajectory: RelationshipTrajectory,
    /// Phi-dyad calculator for relational Phi.
    dyad_calculator: PhiDyadCalculator,
    /// Recent AI states for dyad computation (ring buffer, max 8).
    pub(super) recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    /// Last computed Phi_dyad — fed back into mind as relational Psi on next cycle.
    pub(super) last_phi_dyad: f64,
}

impl RelationalCore {
    pub(super) fn new() -> Self {
        Self {
            partner: HumanPartnerModel::new("human"),
            trajectory: RelationshipTrajectory::default(),
            dyad_calculator: PhiDyadCalculator::new(),
            recent_ai_states: Vec::new(),
            last_phi_dyad: 0.0,
        }
    }

    pub(super) fn from_persisted(
        partner: HumanPartnerModel,
        trajectory: RelationshipTrajectory,
        recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    ) -> Self {
        Self {
            partner,
            trajectory,
            dyad_calculator: PhiDyadCalculator::new(),
            recent_ai_states,
            last_phi_dyad: 0.0,
        }
    }

    /// Push an AI state into the ring buffer (max 8).
    pub(super) fn push_ai_state(&mut self, hv: symthaea_core::hdc::unified_hv::ContinuousHV) {
        self.recent_ai_states.push(hv);
        if self.recent_ai_states.len() > 8 {
            self.recent_ai_states.remove(0);
        }
    }

    /// Compute Phi-dyad from recent AI states and partner model.
    fn compute_phi_dyad(&self) -> f64 {
        if self.recent_ai_states.is_empty() {
            return 0.0;
        }

        let human_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV> = self
            .recent_ai_states
            .iter()
            .map(|s| {
                let mut vals = s.values.clone();
                for v in vals.iter_mut() {
                    *v *= 0.9;
                    *v += 0.1;
                }
                symthaea_core::hdc::unified_hv::ContinuousHV::from_values(vals).normalize()
            })
            .collect();

        let assessment = RelationalAssessment {
            agent_a: "symthaea".to_string(),
            agent_b: self.partner.partner_id.clone(),
            phi_relation: self.partner.phi_relational,
            stage: self.partner.stage,
            synchrony: self.partner.trust as f64,
            turn_taking_quality: 0.7,
            mutual_information: self.partner.reciprocity as f64,
            mode: self.partner.mode,
            num_interactions: self.partner.interactions_count as usize,
            relationship_age: 0.0,
            explanation: String::new(),
        };

        let input = DyadInput {
            ai_states: &self.recent_ai_states,
            human_states: &human_states,
            relational: &assessment,
            human_model: &self.partner,
            weights: DyadWeights::default(),
        };

        self.dyad_calculator.compute(&input).phi_dyad
    }

    /// Update partnership state from interaction consciousness level.
    pub(super) fn update_partnership(&mut self, consciousness: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let depth = (consciousness * 0.5).clamp(0.0, 1.0);
        let safety = (consciousness * 0.7 + 0.2).clamp(0.0, 1.0);
        let mutuality = (consciousness * 0.4 + 0.1).clamp(0.0, 1.0);

        let event = InteractionEvent {
            timestamp: now,
            depth,
            emotional_safety: safety,
            mutuality,
        };
        self.partner.update_on_interaction(&event);

        let assessment = RelationalAssessment {
            agent_a: "symthaea".to_string(),
            agent_b: self.partner.partner_id.clone(),
            phi_relation: self.partner.phi_relational,
            stage: self.partner.stage,
            synchrony: consciousness as f64 * 0.8,
            turn_taking_quality: 0.7,
            mutual_information: mutuality as f64,
            mode: if self.partner.trust > 0.3 {
                RelationMode::IThou
            } else {
                RelationMode::IIt
            },
            num_interactions: self.partner.interactions_count as usize,
            relationship_age: now,
            explanation: String::new(),
        };
        self.partner.update_from_assessment(&assessment);
        self.partner.advance_stage_if_ready();

        let phi_dyad = self.compute_phi_dyad();
        self.trajectory.record(now, self.partner.stage, phi_dyad);
        self.last_phi_dyad = phi_dyad;
    }

    /// Get current partnership state summary.
    pub(super) fn partnership_state(&self) -> PartnershipState {
        let phi_dyad = self.compute_phi_dyad();
        PartnershipState {
            stage: self.partner.stage,
            trust: self.partner.trust,
            vulnerability: self.partner.vulnerability,
            reciprocity: self.partner.reciprocity,
            phi_dyad,
            interactions: self.partner.interactions_count,
            trajectory_points: self.trajectory.points().len(),
        }
    }
}

/// Summary of partnership state for external consumers.
#[derive(Debug, Clone)]
pub struct PartnershipState {
    /// Current relationship stage.
    pub stage: RelationshipStage,
    /// Trust level (0.0-1.0).
    pub trust: f32,
    /// Vulnerability level (0.0-1.0).
    pub vulnerability: f32,
    /// Reciprocity level (0.0-1.0).
    pub reciprocity: f32,
    /// Current Phi-dyad value.
    pub phi_dyad: f64,
    /// Total interactions.
    pub interactions: u64,
    /// Number of trajectory points recorded.
    pub trajectory_points: usize,
}

/// Serializable state for pause/resume persistence.
///
/// Stores relational state (partnership, trajectory), the user-state-inference
/// snapshot, and configuration. The mind and language cores are ephemeral and
/// rebuilt on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct PersistedState {
    pub(super) hdc_dim: usize,
    pub(super) ltc_neurons: usize,
    pub(super) interactions: u64,
    pub(super) partner: HumanPartnerModel,
    pub(super) trajectory: RelationshipTrajectory,
    pub(super) recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    /// Path to the consciousness database (if configured).
    #[serde(default)]
    pub(super) database_path: Option<String>,
    /// Snapshot of Phase 6.5's text/behavior-driven user-state inference
    /// (frustration, cognitive load, experience, engagement). `None` for
    /// state files written before this field existed.
    #[serde(default)]
    pub(super) user_state: Option<crate::user_state_inference::UserState>,
}
