// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Receptor subtypes within a transmitter channel.
///
/// DA: D1 (excitatory/Go pathway → learning) vs D2 (inhibitory/NoGo pathway → flexibility)
/// NE: Alpha (tonic precision/focus) vs Beta (phasic reactivity/startle)
///
/// Science: Frank (2005) — D1/D2 basal ganglia; Arnsten (2000) — alpha/beta NE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptorSubtypes {
    /// Excitatory subtype sensitivity: D1 for DA, Alpha for NE [0.5, 2.0]
    pub excitatory: f32,
    /// Inhibitory subtype sensitivity: D2 for DA, Beta for NE [0.5, 2.0]
    pub inhibitory: f32,
}

impl Default for ReceptorSubtypes {
    fn default() -> Self {
        Self {
            excitatory: 1.0,
            inhibitory: 1.0,
        }
    }
}

/// Signals from the cognitive cycle that drive transmitter production.
pub struct NeuromodulatorInputs {
    /// Prediction error magnitude (0.0–1.0+)
    pub prediction_error: f32,
    /// FEP surprise triggered this cycle
    pub surprise: bool,
    /// Goal achievement / moral alignment (-1.0 to 1.0)
    pub reward_signal: f32,
    /// Temporal coherence (0.0–1.0)
    pub coherence: f32,
    /// Current emotional arousal (0.0–1.0)
    pub arousal: f32,
    /// Phenomenal binding strength (0.0–1.0)
    pub binding_strength: f32,
    /// Epistemic gate confidence (0.0–1.0)
    pub epistemic_confidence: f32,
    /// Whether the system is in flow state
    pub flow_active: bool,
    /// Unified Psi (consciousness level) from previous cycle (0.0–1.0).
    /// Feeds back to modulate 5-HT/DA baselines: high integration → mild boost.
    /// Science: Dehaene et al. (2006) — global workspace access boosts monoaminergic tone.
    pub consciousness_level: Option<f32>,
    /// Moral judgment score from ethics engine (-1.0 to 1.0).
    /// Positive moral actions boost oxytocin (prosocial) and DA (intrinsic reward).
    /// Science: Zak (2012) — oxytocin mediates moral sentiment.
    pub moral_signal: Option<f32>,
}
