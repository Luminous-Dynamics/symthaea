// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Relational consciousness subsystem — partnership, trajectory, and dyadic Phi.

use serde::{Deserialize, Serialize};

use crate::hdc::relational_consciousness::{RelationMode, RelationalAssessment};
use crate::partnership::{HumanPartnerModel, InteractionEvent, RelationshipTrajectory};

pub use crate::hdc::relational_consciousness::RelationshipStage;

/// Relational consciousness subsystem — partnership, trajectory, and dyadic Phi.
///
/// Groups relational state for the facade.
///
/// Important: this facade currently receives user-input embeddings, but no
/// independently sourced partner-observation stream. It therefore does not
/// compute or feed back a dyadic Φ estimate. The standalone PhiDyadCalculator
/// remains available to callers that supply separately sourced streams.
pub(super) struct RelationalCore {
    /// Human partner model for relational consciousness.
    pub(super) partner: HumanPartnerModel,
    /// Relationship trajectory tracking.
    pub(super) trajectory: RelationshipTrajectory,
    /// Recent user utterance embeddings (ring buffer, max 8). These are not
    /// AI-generated states or observations of the user's internal mental state.
    pub(super) recent_user_input_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
}

impl RelationalCore {
    pub(super) fn new() -> Self {
        Self {
            partner: HumanPartnerModel::new("human"),
            trajectory: RelationshipTrajectory::default(),
            recent_user_input_states: Vec::new(),
        }
    }

    pub(super) fn from_persisted(
        partner: HumanPartnerModel,
        trajectory: RelationshipTrajectory,
        recent_user_input_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    ) -> Self {
        Self {
            partner,
            trajectory,
            recent_user_input_states,
            last_phi_dyad: 0.0,
        }
    }

    /// Record a user utterance embedding. This is communication input, not
    /// an observation of the user's private mental state or an AI-generated state.
    pub(super) fn push_user_input_state(
        &mut self,
        hv: symthaea_core::hdc::unified_hv::ContinuousHV,
    ) {
        self.recent_user_input_states.push(hv);
        if self.recent_user_input_states.len() > 8 {
            self.recent_user_input_states.remove(0);
        }
    }

    /// Return a dyadic estimate only when the facade has independent partner
    /// observations. No such stream is currently wired into this subsystem, so
    /// the correct result is unavailable, not a synthetic state or numeric zero.
    fn compute_phi_dyad(&self) -> Option<f64> {
        let _has_user_input_history = !self.recent_user_input_states.is_empty();
        None
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

        if let Some(phi_dyad) = self.compute_phi_dyad() {
            self.trajectory.record(now, self.partner.stage, phi_dyad);
        }
        // When no independently sourced partner stream is available, do not
        // persist an unavailable estimate as a zero-valued trajectory point.
    }

    /// Get current partnership state summary.
    pub(super) fn partnership_state(&self) -> PartnershipState {
        let (phi_dyad, phi_dyad_status) = match self.compute_phi_dyad() {
            Some(value) => (value, PhiDyadStatus::AvailableFromIndependentPartnerObservations),
            None => (0.0, PhiDyadStatus::UnavailableNoIndependentPartnerObservations),
        };
        PartnershipState {
            stage: self.partner.stage,
            trust: self.partner.trust,
            vulnerability: self.partner.vulnerability,
            reciprocity: self.partner.reciprocity,
            phi_dyad,
            phi_dyad_status,
            interactions: self.partner.interactions_count,
            trajectory_points: self.trajectory.points().len(),
        }
    }
}

/// Whether the facade has sufficient independently sourced partner data
/// to report a dyadic Φ estimate. This is a data-availability/provenance status,
/// not a judgement about a relationship or a claim about consciousness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiDyadStatus {
    /// The current facade does not receive independent partner observations.
    UnavailableNoIndependentPartnerObservations,
    /// Reserved for a future provenance-validated observation path.
    AvailableFromIndependentPartnerObservations,
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
    /// Current Phi-dyad value. Zero is a compatibility placeholder when the
    /// status says the estimate is unavailable; do not interpret it as an
    /// observed zero-valued Φ measurement.
    pub phi_dyad: f64,
    /// Explicitly distinguishes unavailable input from a measured value.
    pub phi_dyad_status: PhiDyadStatus,
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
    /// Legacy serialized key "recent_ai_states" held user-input embeddings in the facade.
    /// Alias preserves old snapshots while the in-memory name reflects their source.
    #[serde(default, alias = "recent_ai_states")]
    pub(super) recent_user_input_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    /// Path to the consciousness database (if configured).
    #[serde(default)]
    pub(super) database_path: Option<String>,
    /// Snapshot of Phase 6.5's text/behavior-driven user-state inference
    /// (frustration, cognitive load, experience, engagement). `None` for
    /// state files written before this field existed.
    #[serde(default)]
    pub(super) user_state: Option<crate::user_state_inference::UserState>,
}


#[cfg(test)]
mod provenance_tests {
    use super::*;

    #[test]
    fn facade_phi_is_unavailable_without_independent_partner_observations() {
        let mut relational = RelationalCore::new();
        let user_input = symthaea_core::hdc::unified_hv::ContinuousHV::from_values(vec![0.25; 16]);
        relational.push_user_input_state(user_input);

        assert_eq!(relational.compute_phi_dyad(), None);

        relational.update_partnership(0.9);
        assert_eq!(
            relational.partnership_state().phi_dyad_status,
            PhiDyadStatus::UnavailableNoIndependentPartnerObservations
        );
        assert!(
            relational.trajectory.points().is_empty(),
            "unavailable Phi must not be persisted as a zero-valued observation"
        );
    }

    #[test]
    fn empty_facade_has_explicitly_unavailable_phi() {
        let relational = RelationalCore::new();
        let state = relational.partnership_state();
        assert_eq!(state.phi_dyad, 0.0);
        assert_eq!(
            state.phi_dyad_status,
            PhiDyadStatus::UnavailableNoIndependentPartnerObservations
        );
    }
}
