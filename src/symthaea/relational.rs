// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Relational consciousness subsystem — partnership, trajectory, and dyadic Phi.

use serde::{Deserialize, Serialize};

use crate::partnership::{HumanPartnerModel, RelationshipTrajectory};

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
        mut partner: HumanPartnerModel,
        _legacy_trajectory: RelationshipTrajectory,
        recent_user_input_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    ) -> Self {
        // Pre-gate facade snapshots stored heuristic relationship fields and
        // ungrounded dyadic-Phi trajectory points without source provenance.
        // Do not promote those legacy values into observed relationship evidence.
        partner.mode = crate::hdc::relational_consciousness::RelationMode::IIt;
        partner.stage = RelationshipStage::NoRelation;
        partner.trust = 0.0;
        partner.vulnerability = 0.0;
        partner.reciprocity = 0.0;
        partner.phi_relational = 0.0;

        Self {
            partner,
            trajectory: RelationshipTrajectory::default(),
            recent_user_input_states,
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

    /// Record that a user turn occurred without inferring their trust,
    /// emotional safety, vulnerability, reciprocity, or relationship stage from
    /// the system's own consciousness-related scalar.
    pub(super) fn record_interaction(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.partner.interactions_count = self.partner.interactions_count.saturating_add(1);
        self.partner.last_interaction_ts = Some(now);

        if let Some(phi_dyad) = self.compute_phi_dyad() {
            self.trajectory.record(now, self.partner.stage, phi_dyad);
        }
        // When no independently sourced partner stream is available, do not
        // persist an unavailable estimate as a zero-valued trajectory point.
    }

    /// Get current partnership state summary.
    pub(super) fn partnership_state(&self) -> PartnershipState {
        let phi_dyad = self.compute_phi_dyad().unwrap_or(0.0);
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

/// Whether the facade has sufficient independently sourced partner data
/// to report a dyadic Φ estimate. This is a data-availability/provenance status,
/// not a judgement about a relationship or a claim about consciousness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiDyadStatus {
    /// The current facade does not receive independent partner observations.
    UnavailableNoIndependentPairedStreams,
    /// Reserved for a future provenance-validated observation path.
    AvailableFromIndependentPartnerObservations,
}

/// Evidence status for the relationship-state fields exposed by the facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartnerStateEvidenceStatus {
    /// No independently sourced partner observations are currently wired in.
    UnavailableNoIndependentPartnerObservations,
    /// Reserved for a future independently sourced observation path.
    AvailableFromIndependentPartnerObservations,
}

/// Summary of partnership state for external consumers.
#[derive(Debug, Clone)]
pub struct PartnershipState {
    /// Current relationship stage.
    pub stage: RelationshipStage,
    /// Trust level (0.0-1.0). See partner_state_status() before interpreting.
    pub trust: f32,
    /// Vulnerability level (0.0-1.0). See partner_state_status() before interpreting.
    pub vulnerability: f32,
    /// Reciprocity level (0.0-1.0). See partner_state_status() before interpreting.
    pub reciprocity: f32,
    /// Current Phi-dyad value. Zero is a compatibility placeholder when
    /// phi_dyad_status() says the estimate is unavailable; do not interpret it
    /// as an observed zero-valued Φ measurement.
    pub phi_dyad: f64,
    /// Total interactions.
    pub interactions: u64,
    /// Number of trajectory points recorded.
    pub trajectory_points: usize,
}

impl PartnershipState {
    /// Evidence status of the partner-state fields. The current facade has no
    /// independent partner-observation input, so its estimates are unavailable.
    pub fn partner_state_status(&self) -> PartnerStateEvidenceStatus {
        PartnerStateEvidenceStatus::UnavailableNoIndependentPartnerObservations
    }

    /// Evidence status of the dyadic Phi estimate. Never infer availability
    /// from the numeric value, since a valid estimate could also equal zero.
    pub fn phi_dyad_status(&self) -> PhiDyadStatus {
        PhiDyadStatus::UnavailableNoIndependentPartnerObservations
    }
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

        relational.record_interaction();
        assert_eq!(relational.partner.trust, 0.0);
        assert_eq!(relational.partner.reciprocity, 0.0);
        assert_eq!(relational.partner.stage, RelationshipStage::NoRelation);
        assert_eq!(
            relational.partnership_state().phi_dyad_status(),
            PhiDyadStatus::UnavailableNoIndependentPartnerObservations
        );
        assert_eq!(
            relational.partnership_state().partner_state_status(),
            PartnerStateEvidenceStatus::UnavailableNoIndependentPartnerObservations
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
            state.phi_dyad_status(),
            PhiDyadStatus::UnavailableNoIndependentPartnerObservations
        );
        assert_eq!(
            state.partner_state_status(),
            PartnerStateEvidenceStatus::UnavailableNoIndependentPartnerObservations
        );
    }

    #[test]
    fn legacy_proxy_state_and_trajectory_are_not_promoted_to_observed_data() {
        let mut legacy_partner = HumanPartnerModel::new("legacy");
        legacy_partner.trust = 0.95;
        legacy_partner.vulnerability = 0.8;
        legacy_partner.reciprocity = 0.9;
        legacy_partner.phi_relational = 0.99;
        legacy_partner.stage = RelationshipStage::Attunement;

        let mut legacy_trajectory = RelationshipTrajectory::default();
        legacy_trajectory.record(1.0, RelationshipStage::Attunement, 0.8);
        let restored = RelationalCore::from_persisted(
            legacy_partner,
            legacy_trajectory,
            Vec::new(),
        );
        let state = restored.partnership_state();

        assert_eq!(state.trust, 0.0);
        assert_eq!(state.vulnerability, 0.0);
        assert_eq!(state.reciprocity, 0.0);
        assert_eq!(state.stage, RelationshipStage::NoRelation);
        assert_eq!(state.phi_dyad, 0.0);
        assert_eq!(state.phi_dyad_status(), PhiDyadStatus::UnavailableNoIndependentPartnerObservations);
        assert_eq!(state.trajectory_points, 0);
    }
}
