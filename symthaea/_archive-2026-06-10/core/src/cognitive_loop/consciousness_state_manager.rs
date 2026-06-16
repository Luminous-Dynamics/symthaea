// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ConsciousnessStateManager: groups advanced consciousness analysis subsystems.
//!
//! Consolidates 4 `Option<T>` fields that collectively implement consciousness
//! measurement via different mechanisms:
//! - Cross-modal binding (Treisman 1996, Engel et al. 2001)
//! - Predictive processing (Friston 2010, Clark 2015)
//! - Phi-guided attention gating (Tononi 2004)
//! - Affective bridge / emotion-cognition coupling (Russell 1980, Damasio 2003)

use crate::attention::PhiAttentionGate;
use crate::brain::affective_bridge::AffectiveBridge;
use crate::consciousness::cross_modal_binding::CrossModalBinder;
use crate::consciousness::predictive_processing::{PredictiveConfig, PredictiveMind};

/// Manager grouping advanced consciousness state subsystems.
///
/// All fields are `Option<T>` — when `None` the subsystem is disabled and the
/// cycle skips its processing. Config flags in `CognitiveLoopConfig` control
/// which subsystems are enabled.
#[derive(Debug)]
pub struct ConsciousnessStateManager {
    /// Cross-modal binder for multi-modal integration.
    /// When enabled, binds HDC encodings across modalities and computes cross-modal Phi.
    pub cross_modal_binder: Option<CrossModalBinder>,

    /// Predictive processing mind (hierarchical predictive coding + precision dynamics).
    /// When enabled, provides phi_modulation from free energy minimization.
    pub predictive_mind: Option<PredictiveMind>,

    /// Phi-guided attention gate for consciousness-aware perception weighting.
    /// When present, weights perception inputs by their integrated information
    /// contribution, focusing processing on high-Phi signals.
    pub phi_attention_gate: Option<PhiAttentionGate>,

    /// Affective bridge for emotion-cognition coupling.
    /// When enabled, evaluates somatic marker signals from cognitive loop state.
    pub affective_bridge: Option<AffectiveBridge>,
}

impl Default for ConsciousnessStateManager {
    fn default() -> Self {
        Self {
            cross_modal_binder: None,
            predictive_mind: None,
            phi_attention_gate: None,
            affective_bridge: None,
        }
    }
}

impl ConsciousnessStateManager {
    /// Reset all subsystems to fresh state (used by `CognitiveLoopService::reset()`).
    pub fn reset(&mut self) {
        if let Some(ref mut mind) = self.predictive_mind {
            *mind = PredictiveMind::new(PredictiveConfig::default());
        }
        if let Some(ref mut binder) = self.cross_modal_binder {
            binder.clear();
        }
        if let Some(ref mut bridge) = self.affective_bridge {
            *bridge = AffectiveBridge::default();
        }
        // phi_attention_gate has no reset — it's stateless (forward-only).
    }
}
