// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Main AttachmentSystem — Bowlby's attachment theory engine.
//!
//! Tracks caregiver interaction history via EMA smoothing, updates internal
//! working models, and produces emergent attachment styles. The attachment
//! relationship is encoded as a 16,384D hypervector that accumulates interaction
//! patterns weighted by critical period plasticity.
//!
//! Science: Bowlby (1969/1982), Ainsworth et al. (1978), Main & Solomon (1990).

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::attachment_style_formation::determine_style;
use crate::critical_periods::{CriticalPeriodDomain, CriticalPeriodModel};
use crate::internal_working_model::{new_working_model, update_working_model};
use crate::separation_distress::advance_separation;
use crate::types::{
    AttachmentNeuromodulation, AttachmentStyle, CaregiverAction, InternalWorkingModel,
    SeparationStage,
};

/// Encode a caregiver interaction as a hypervector.
///
/// Uses deterministic basis vectors for each dimension, scaled by the
/// action's quality metrics, then bundled and normalized.
pub fn encode_interaction(action: &CaregiverAction) -> ContinuousHV {
    let resp_basis = ContinuousHV::random(HDC_DIMENSION, 2001);
    let warm_basis = ContinuousHV::random(HDC_DIMENSION, 2002);
    let attn_basis = ContinuousHV::random(HDC_DIMENSION, 2003);
    let contact_basis = ContinuousHV::random(HDC_DIMENSION, 2004);
    let vocal_basis = ContinuousHV::random(HDC_DIMENSION, 2005);

    let resp_hv = resp_basis.scale(action.responsiveness as f32);
    let warm_hv = warm_basis.scale(action.warmth as f32);
    let attn_hv = attn_basis.scale(action.attunement as f32);
    let contact_hv = contact_basis.scale(if action.contact { 1.0 } else { 0.0 });
    let vocal_hv = vocal_basis.scale(if action.vocalizing { 1.0 } else { 0.0 });

    ContinuousHV::bundle(&[&resp_hv, &warm_hv, &attn_hv, &contact_hv, &vocal_hv]).normalize()
}

/// Core attachment system tracking the caregiver-infant relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentSystem {
    /// Current attachment style (emergent).
    pub style: AttachmentStyle,
    /// Composite security score (0-1).
    pub security_score: f64,
    /// P(respond | distress), learned via EMA.
    pub caregiver_reliability: f64,
    /// EMA of caregiver response latency (seconds).
    pub response_latency_ema: f64,
    /// EMA of response quality (warmth/appropriateness).
    pub response_quality_ema: f64,
    /// Current separation distress level (0-1).
    pub separation_distress_level: f64,
    /// Current stage of separation distress cascade.
    pub separation_stage: SeparationStage,
    /// Proximity-seeking drive intensity (0-1).
    pub proximity_seeking_drive: f64,
    /// Internal working model (self, other, world beliefs).
    pub internal_working_model: InternalWorkingModel,
    /// Critical period plasticity (1.0 at birth, decays with age).
    pub critical_period_plasticity: f64,
    /// 16,384D hypervector encoding of attachment relationship.
    pub attachment_hv: ContinuousHV,
    /// Total interactions processed.
    pub interaction_count: u64,
    /// Current developmental age in months.
    pub total_months: f64,
    /// Whether the caregiver has exhibited frightening behavior.
    pub frightening_history: bool,
    /// Separation duration accumulator (minutes).
    pub separation_minutes: f64,
    /// Critical period model for plasticity computation.
    #[serde(skip)]
    critical_period_model: Option<CriticalPeriodModel>,
}

impl AttachmentSystem {
    /// Create a new attachment system at birth (0 months).
    pub fn new() -> Self {
        Self {
            style: AttachmentStyle::Forming,
            security_score: 0.5,
            caregiver_reliability: 0.5,
            response_latency_ema: 30.0,
            response_quality_ema: 0.5,
            separation_distress_level: 0.0,
            separation_stage: SeparationStage::Calm,
            proximity_seeking_drive: 0.3,
            internal_working_model: new_working_model(),
            critical_period_plasticity: 1.0,
            attachment_hv: ContinuousHV::random(HDC_DIMENSION, 42),
            interaction_count: 0,
            total_months: 0.0,
            frightening_history: false,
            separation_minutes: 0.0,
            critical_period_model: Some(CriticalPeriodModel::new()),
        }
    }

    /// Process a single caregiver interaction.
    ///
    /// Updates EMAs, reliability estimates, internal working model, attachment
    /// HV, and recomputes style. The learning rate is modulated by critical
    /// period plasticity.
    pub fn process_interaction(&mut self, action: &CaregiverAction) {
        self.interaction_count += 1;

        // EMA smoothing factor
        let alpha = 0.05;

        // Update quality EMA
        self.response_quality_ema =
            alpha * action.quality() + (1.0 - alpha) * self.response_quality_ema;

        // Update reliability EMA
        self.caregiver_reliability =
            alpha * action.responsiveness + (1.0 - alpha) * self.caregiver_reliability;

        // Track frightening behavior
        if action.warmth < 0.05 && action.contact {
            self.frightening_history = true;
        }

        // Update internal working model (plasticity-modulated)
        update_working_model(
            &mut self.internal_working_model,
            action,
            self.critical_period_plasticity,
        );

        // Update security score (composite)
        self.security_score = 0.4 * self.caregiver_reliability
            + 0.3 * self.response_quality_ema
            + 0.3 * self.internal_working_model.other_reliable;

        // Update attachment HV (accumulate interaction, weighted by plasticity)
        let interaction_hv = encode_interaction(action);
        self.attachment_hv = self
            .attachment_hv
            .add(&interaction_hv.scale(self.critical_period_plasticity as f32))
            .normalize();

        // Reset separation state on interaction
        if action.responsiveness > 0.3 {
            self.separation_stage = SeparationStage::Calm;
            self.separation_distress_level = 0.0;
            self.separation_minutes = 0.0;
        }

        // Recompute style if enough interactions
        if self.interaction_count >= 20 {
            self.update_style();
        }
    }

    /// Process a separation event (caregiver absent for `duration_minutes`).
    pub fn process_separation(&mut self, duration_minutes: f64) {
        self.separation_minutes += duration_minutes;

        // Advance separation distress stage
        self.separation_stage =
            advance_separation(self.separation_stage, self.separation_minutes, self);

        self.separation_distress_level = self.separation_stage.distress_level();

        // Update proximity-seeking drive
        self.proximity_seeking_drive = match self.style {
            AttachmentStyle::Anxious => (0.5 + 0.5 * self.separation_distress_level).min(1.0),
            AttachmentStyle::Avoidant => 0.2, // Suppressed
            AttachmentStyle::Disorganized => 0.3 + 0.4 * self.separation_distress_level,
            _ => 0.3 + 0.3 * self.separation_distress_level,
        };
    }

    /// Process a reunion with caregiver after separation.
    pub fn process_reunion(&mut self, action: &CaregiverAction) -> AttachmentNeuromodulation {
        let was_distressed = self.separation_distress_level > 0.3;

        // Reset separation
        self.separation_stage = SeparationStage::Calm;
        self.separation_distress_level = 0.0;
        self.separation_minutes = 0.0;
        self.proximity_seeking_drive = (self.proximity_seeking_drive - 0.2).max(0.0);

        // Process as regular interaction
        self.process_interaction(action);

        // Return neuromodulation based on reunion context
        if was_distressed {
            AttachmentNeuromodulation::reunion()
        } else {
            AttachmentNeuromodulation::coregulation(action.quality())
        }
    }

    /// Update attachment style based on accumulated interaction history.
    pub fn update_style(&mut self) {
        self.style = determine_style(
            self.caregiver_reliability,
            self.response_quality_ema,
            self.response_quality_ema, // warmth proxy
            self.frightening_history,
            self.total_months,
        );
    }

    /// Advance developmental age and update critical period plasticity.
    pub fn advance_age(&mut self, months: f64) {
        self.total_months += months;

        // Update critical period plasticity
        let model = self
            .critical_period_model
            .get_or_insert_with(CriticalPeriodModel::new);
        let age = crate::types::DevelopmentalAge::new(self.total_months);
        self.critical_period_plasticity =
            model.plasticity_at(CriticalPeriodDomain::Attachment, &age);
    }

    /// Get the neuromodulatory signature for the current separation state.
    pub fn current_neuromodulation(&self) -> AttachmentNeuromodulation {
        crate::separation_distress::neuromod_for_stage(self.separation_stage)
    }
}

impl Default for AttachmentSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_attachment_system() {
        let sys = AttachmentSystem::new();
        assert_eq!(sys.style, AttachmentStyle::Forming);
        assert_eq!(sys.interaction_count, 0);
        assert!((sys.total_months - 0.0).abs() < 1e-10);
        assert!((sys.critical_period_plasticity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reliable_caregiver_produces_secure_attachment() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(8.0); // Past forming stage

        let reliable = CaregiverAction::reliable();
        for _ in 0..1000 {
            sys.process_interaction(&reliable);
        }

        assert_eq!(
            sys.style,
            AttachmentStyle::Secure,
            "Reliable caregiver should produce secure attachment, got {:?}",
            sys.style
        );
        assert!(
            sys.security_score > 0.7,
            "Security score should be high, got {}",
            sys.security_score
        );
        assert!(
            sys.internal_working_model.self_worthy > 0.7,
            "Self-worthy should be high with warm caregiver, got {}",
            sys.internal_working_model.self_worthy
        );
    }

    #[test]
    fn test_inconsistent_caregiver_produces_anxious_attachment() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(8.0);

        let inconsistent = CaregiverAction::inconsistent();
        for _ in 0..1000 {
            sys.process_interaction(&inconsistent);
        }

        assert_eq!(
            sys.style,
            AttachmentStyle::Anxious,
            "Inconsistent caregiver should produce anxious attachment, got {:?}",
            sys.style
        );
    }

    #[test]
    fn test_unresponsive_caregiver_produces_avoidant_attachment() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(8.0);

        let unresponsive = CaregiverAction::unresponsive();
        for _ in 0..1000 {
            sys.process_interaction(&unresponsive);
        }

        assert_eq!(
            sys.style,
            AttachmentStyle::Avoidant,
            "Unresponsive caregiver should produce avoidant attachment, got {:?}",
            sys.style
        );
    }

    #[test]
    fn test_frightening_caregiver_produces_disorganized_attachment() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(8.0);

        let frightening = CaregiverAction::frightening();
        for _ in 0..1000 {
            sys.process_interaction(&frightening);
        }

        assert_eq!(
            sys.style,
            AttachmentStyle::Disorganized,
            "Frightening caregiver should produce disorganized attachment, got {:?}",
            sys.style
        );
    }

    #[test]
    fn test_security_score_increases_with_reliable_care() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(6.0);

        let initial_score = sys.security_score;
        let reliable = CaregiverAction::reliable();
        for _ in 0..100 {
            sys.process_interaction(&reliable);
        }

        assert!(
            sys.security_score > initial_score,
            "Security score should increase with reliable care: {} -> {}",
            initial_score,
            sys.security_score
        );
    }

    #[test]
    fn test_security_score_decreases_with_neglect() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(6.0);

        // First establish moderate baseline
        let reliable = CaregiverAction::reliable();
        for _ in 0..200 {
            sys.process_interaction(&reliable);
        }
        let baseline = sys.security_score;

        // Then switch to neglect
        let unresponsive = CaregiverAction::unresponsive();
        for _ in 0..500 {
            sys.process_interaction(&unresponsive);
        }

        assert!(
            sys.security_score < baseline,
            "Security score should decrease with neglect: {} -> {}",
            baseline,
            sys.security_score
        );
    }

    #[test]
    fn test_forming_before_six_months() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(3.0); // Only 3 months

        let reliable = CaregiverAction::reliable();
        for _ in 0..100 {
            sys.process_interaction(&reliable);
        }

        assert_eq!(
            sys.style,
            AttachmentStyle::Forming,
            "Should still be Forming before 6 months"
        );
    }

    #[test]
    fn test_separation_distress_cascade() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(12.0);

        // Establish some attachment first
        let reliable = CaregiverAction::reliable();
        for _ in 0..100 {
            sys.process_interaction(&reliable);
        }

        // Now separate
        sys.process_separation(3.0);
        assert_eq!(sys.separation_stage, SeparationStage::Calm);

        sys.process_separation(3.0); // Total 6 min
        assert_eq!(sys.separation_stage, SeparationStage::Quiet);

        sys.process_separation(10.0); // Total 16 min
        assert_eq!(sys.separation_stage, SeparationStage::Vocalization);

        sys.process_separation(15.0); // Total 31 min
        assert_eq!(sys.separation_stage, SeparationStage::Protest);

        sys.process_separation(30.0); // Total 61 min
        assert_eq!(sys.separation_stage, SeparationStage::Despair);
    }

    #[test]
    fn test_reunion_resets_distress() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(12.0);

        // Build attachment, then separate
        let reliable = CaregiverAction::reliable();
        for _ in 0..100 {
            sys.process_interaction(&reliable);
        }
        // Advance through separation stages (each call advances one stage)
        sys.process_separation(10.0); // Calm → Quiet (>5 min)
        sys.process_separation(10.0); // Quiet → Vocalization (>15 min cumulative)
        sys.process_separation(20.0); // Vocalization → Protest (>30 min cumulative)
        assert!(
            sys.separation_distress_level > 0.3,
            "After protest-level separation, distress should be > 0.3, got {}",
            sys.separation_distress_level
        );

        // Reunion
        let neuromod = sys.process_reunion(&reliable);
        assert_eq!(sys.separation_stage, SeparationStage::Calm);
        assert!((sys.separation_distress_level - 0.0).abs() < 1e-10);
        assert!(
            neuromod.oxytocin_burst > 0.0,
            "Reunion should produce oxytocin burst"
        );
    }

    #[test]
    fn test_advance_age_reduces_plasticity() {
        let mut sys = AttachmentSystem::new();
        assert!((sys.critical_period_plasticity - 1.0).abs() < 1e-10);

        sys.advance_age(30.0); // Past critical period
        assert!(
            sys.critical_period_plasticity < 0.8,
            "Plasticity should decrease after 24mo, got {}",
            sys.critical_period_plasticity
        );
    }

    #[test]
    fn test_attachment_hv_evolves() {
        let mut sys = AttachmentSystem::new();
        let initial_hv = sys.attachment_hv.clone();

        let reliable = CaregiverAction::reliable();
        for _ in 0..50 {
            sys.process_interaction(&reliable);
        }

        let sim = sys.attachment_hv.similarity(&initial_hv);
        assert!(
            sim < 0.99,
            "Attachment HV should evolve with interactions, sim={sim}"
        );
    }

    #[test]
    fn test_interaction_count_increments() {
        let mut sys = AttachmentSystem::new();
        let action = CaregiverAction::reliable();
        for _ in 0..37 {
            sys.process_interaction(&action);
        }
        assert_eq!(sys.interaction_count, 37);
    }

    #[test]
    fn test_proximity_seeking_anxious_vs_avoidant() {
        // Anxious should have high proximity seeking during separation
        let mut anxious = AttachmentSystem::new();
        anxious.advance_age(12.0);
        for _ in 0..500 {
            anxious.process_interaction(&CaregiverAction::inconsistent());
        }
        anxious.process_separation(40.0);

        // Avoidant should have low proximity seeking during separation
        let mut avoidant = AttachmentSystem::new();
        avoidant.advance_age(12.0);
        for _ in 0..500 {
            avoidant.process_interaction(&CaregiverAction::unresponsive());
        }
        avoidant.process_separation(40.0);

        assert!(
            anxious.proximity_seeking_drive > avoidant.proximity_seeking_drive,
            "Anxious should seek proximity more than avoidant: {} vs {}",
            anxious.proximity_seeking_drive,
            avoidant.proximity_seeking_drive
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(12.0);
        let reliable = CaregiverAction::reliable();
        for _ in 0..50 {
            sys.process_interaction(&reliable);
        }

        let json = serde_json::to_string(&sys).unwrap();
        let deserialized: AttachmentSystem = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.style, sys.style);
        assert_eq!(deserialized.interaction_count, sys.interaction_count);
        assert!((deserialized.security_score - sys.security_score).abs() < 1e-10);
    }
}
