// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social Manager — groups relational, partnership, and social coherence state.
//!
//! Consolidates 6 previously scattered social/relational fields from
//! CognitiveLoopService into a single coherent module.

use crate::brain::social_coherence::{InteractionType, SocialCoherence, SocialCoherenceConfig};
use crate::partnership::{HumanPartnerModel, PhiDyadCalculator};
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_pragmatics::speech_act::{SpeechAct, classify};

/// Social/external signal state.
pub(crate) struct SocialState {
    /// Relational Psi from dyad computation (15% blend weight into unified_psi).
    pub relational_psi: f64,
    /// External reward signal injected by environment (0.0 = none).
    pub external_reward: f32,
    /// Social trust level injected by Mind module's SocialCoherence (0.0–1.0).
    pub social_trust: f32,
    /// Social cooperation rate injected by Mind module's SocialCoherence (0.0–1.0).
    pub social_cooperation_rate: f32,
    /// Rolling accuracy of social predictions (0.0–1.0).
    pub social_prediction_accuracy: f32,
    /// Number of active mental models being tracked by SocialCoherence.
    pub social_models_count: usize,
    /// Mean trust across all tracked relationships.
    pub social_mean_trust: f32,
}

impl Default for SocialState {
    fn default() -> Self {
        Self {
            relational_psi: 0.0,
            external_reward: 0.0,
            social_trust: 0.5,
            social_cooperation_rate: 0.0,
            social_prediction_accuracy: 0.5,
            social_models_count: 0,
            social_mean_trust: 0.5,
        }
    }
}

/// Consolidated social/relational manager.
///
/// Groups partnership tracking, phi-dyad computation, and social signal
/// state into a single struct. Replaces the old `SocialState` +
/// `SocialCoherenceState` + 4 separate fields pattern.
pub(crate) struct SocialManager {
    /// Social signal state (trust, cooperation, reward, relational psi).
    pub social: SocialState,

    /// Phi-Dyad calculator for relational consciousness.
    pub phi_dyad: Option<PhiDyadCalculator>,

    /// Human partner model for relational state tracking.
    pub partner_model: Option<HumanPartnerModel>,

    /// Social Coherence: Theory of Mind and social reasoning engine.
    pub coherence: Option<SocialCoherence>,

    /// Ring buffer of recent AI HDC states (last 4, for dyad computation).
    pub recent_ai_hvs: Vec<ContinuousHV>,

    /// Ring buffer of recent input HDC states (last 4, as human proxy).
    pub recent_input_hvs: Vec<ContinuousHV>,
}

impl Default for SocialManager {
    fn default() -> Self {
        Self {
            social: SocialState::default(),
            phi_dyad: None,
            partner_model: None,
            coherence: None,
            recent_ai_hvs: Vec::with_capacity(4),
            recent_input_hvs: Vec::with_capacity(4),
        }
    }
}

impl SocialManager {
    /// Create a new SocialManager with optional phi-dyad and partner model.
    pub fn new(enable_social: bool) -> Self {
        Self {
            social: SocialState::default(),
            phi_dyad: if enable_social {
                Some(PhiDyadCalculator::new())
            } else {
                None
            },
            partner_model: if enable_social {
                Some(HumanPartnerModel::new("human"))
            } else {
                None
            },
            coherence: if enable_social {
                Some(SocialCoherence::new(SocialCoherenceConfig::default()))
            } else {
                None
            },
            recent_ai_hvs: Vec::with_capacity(4),
            recent_input_hvs: Vec::with_capacity(4),
        }
    }

    /// Sync metrics from SocialCoherence into SocialState.
    pub fn sync_coherence_metrics(&mut self) {
        if let Some(ref sc) = self.coherence {
            let stats = sc.stats();
            self.social.social_trust = stats.avg_trust;
            self.social.social_cooperation_rate = stats.cooperation_rate;
            self.social.social_models_count = sc.get_allies().len() + sc.get_rivals().len(); // Approximate
            self.social.social_mean_trust = stats.avg_trust;
            if stats.total_predictions > 0 {
                self.social.social_prediction_accuracy =
                    stats.successful_predictions as f32 / stats.total_predictions as f32;
            }
        }
    }

    /// Record a social observation of another agent.
    pub fn record_observation(
        &mut self,
        agent_id: &str,
        behavior: &ContinuousHV,
        context: &ContinuousHV,
    ) {
        if let Some(ref mut sc) = self.coherence {
            sc.observe_agent(agent_id, behavior, context);
        }
    }

    /// Record a direct interaction with another agent.
    pub fn record_interaction(
        &mut self,
        agent_id: &str,
        outcome: f32,
        context: &ContinuousHV,
        our_action: &str,
        their_response: &str,
    ) {
        if let Some(ref mut sc) = self.coherence {
            let interaction_type = classify_interaction(outcome, their_response);
            sc.record_interaction(
                agent_id,
                interaction_type,
                outcome,
                context.clone(),
                our_action,
                their_response,
            );
        }
    }
}

/// Classify a social interaction, enriching the coarse outcome-sign signal with
/// the *speech act* of the other agent's response (`symthaea-pragmatics`).
///
/// A promise (commissive) reads as cooperation regardless of the immediate
/// outcome; a request/command/question (directive) is negotiation in good faith
/// but competition under a negative outcome; positive affect (expressive) is
/// help. Plain statements and performatives (assertive / declarative — also the
/// default for empty or neutral text) keep the original outcome-sign behaviour,
/// so this is additive: the pre-pragmatics mapping is unchanged for that case.
fn classify_interaction(outcome: f32, their_response: &str) -> InteractionType {
    let base = if outcome >= 0.0 {
        InteractionType::Cooperation
    } else {
        InteractionType::Conflict
    };
    match classify(their_response) {
        SpeechAct::Commissive => InteractionType::Cooperation,
        SpeechAct::Directive => {
            if outcome >= 0.0 {
                InteractionType::Negotiation
            } else {
                InteractionType::Competition
            }
        }
        SpeechAct::Expressive => {
            if outcome >= 0.0 {
                InteractionType::Help
            } else {
                InteractionType::Conflict
            }
        }
        SpeechAct::Assertive | SpeechAct::Declarative => base,
    }
}

#[cfg(test)]
mod tests {
    use super::classify_interaction;
    use crate::brain::social_coherence::InteractionType;

    #[test]
    fn neutral_text_keeps_outcome_sign_default() {
        // Plain assertion / empty text → unchanged pre-pragmatics behaviour.
        assert_eq!(
            classify_interaction(0.5, "the sky is blue"),
            InteractionType::Cooperation
        );
        assert_eq!(
            classify_interaction(-0.5, "the sky is blue"),
            InteractionType::Conflict
        );
        assert_eq!(classify_interaction(0.2, ""), InteractionType::Cooperation);
    }

    #[test]
    fn promise_is_cooperation_even_if_outcome_negative() {
        assert_eq!(
            classify_interaction(-0.3, "I promise to help you rebuild it"),
            InteractionType::Cooperation
        );
    }

    #[test]
    fn directive_splits_on_outcome() {
        assert_eq!(
            classify_interaction(0.4, "please send the report?"),
            InteractionType::Negotiation
        );
        assert_eq!(
            classify_interaction(-0.4, "give me the report now"),
            InteractionType::Competition
        );
    }

    #[test]
    fn positive_affect_is_help() {
        assert_eq!(
            classify_interaction(0.6, "thank you so much for this"),
            InteractionType::Help
        );
    }
}
