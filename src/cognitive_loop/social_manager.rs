// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social Manager — groups relational, partnership, and social coherence state.
//!
//! Consolidates 6 previously scattered social/relational fields from
//! CognitiveLoopService into a single coherent module.

use crate::partnership::{HumanPartnerModel, PhiDyadCalculator};
use symthaea_core::hdc::unified_hv::ContinuousHV;

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
            recent_ai_hvs: Vec::with_capacity(4),
            recent_input_hvs: Vec::with_capacity(4),
        }
    }
}

impl SocialManager {
    /// Create a new SocialManager with optional phi-dyad and partner model.
    pub fn new(enable_phi_dyad: bool) -> Self {
        Self {
            social: SocialState::default(),
            phi_dyad: if enable_phi_dyad {
                Some(PhiDyadCalculator::new())
            } else {
                None
            },
            partner_model: if enable_phi_dyad {
                Some(HumanPartnerModel::new("human"))
            } else {
                None
            },
            recent_ai_hvs: Vec::with_capacity(4),
            recent_input_hvs: Vec::with_capacity(4),
        }
    }
}
