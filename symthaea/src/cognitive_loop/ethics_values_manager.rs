// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EthicsAndValuesManager: groups ethics, values, and moral reasoning subsystems.
//!
//! Consolidates 5 fields that collectively implement ethical evaluation
//! and value alignment:
//! - Last moral judgment tracking (for learning + telemetry)
//! - Contextual harmony weights (domain-aware ethical reasoning)
//! - Phi-attention routing (adaptive thresholds for attention allocation)
//! - Negation detection (moral/value text preprocessing)
//! - Soul (Eight Harmonies value alignment + long-term value learning)

use super::MoralJudgmentSummary;

/// Manager grouping ethics, values, and moral reasoning subsystems.
///
/// All optional fields are `Option<T>` — when `None` the subsystem is
/// disabled and the cycle skips its processing. Config flags in
/// `CognitiveLoopConfig` control which subsystems are enabled.
#[derive(Debug)]
pub struct EthicsAndValuesManager {
    /// Last moral evaluation result (for tracking and learning).
    pub last_moral_judgment: Option<MoralJudgmentSummary>,

    /// Contextual harmony weighting for domain-aware ethical reasoning.
    pub contextual_weights: Option<crate::consciousness::contextual_weights::ContextualWeights>,

    /// Phi-weighted attention routing with adaptive thresholds.
    pub phi_attention: Option<crate::consciousness::phi_attention::AdaptiveThresholds>,

    /// Negation detector for moral/value text preprocessing.
    pub negation_detector: Option<crate::consciousness::negation_detector::NegationDetector>,

    /// Soul: Eight Harmonies value alignment for moral evaluation.
    pub soul: Option<crate::soul::Soul>,
}

impl Default for EthicsAndValuesManager {
    fn default() -> Self {
        Self {
            last_moral_judgment: None,
            contextual_weights: None,
            phi_attention: None,
            negation_detector: None,
            soul: None,
        }
    }
}

impl EthicsAndValuesManager {
    /// Reset all subsystems to fresh state (used by `CognitiveLoopService::reset()`).
    pub fn reset(&mut self) {
        self.last_moral_judgment = None;
        if let Some(ref mut cw) = self.contextual_weights {
            *cw = crate::consciousness::contextual_weights::ContextualWeights::new();
        }
        if let Some(ref mut pa) = self.phi_attention {
            *pa = crate::consciousness::phi_attention::AdaptiveThresholds::new(100);
        }
        if let Some(ref mut nd) = self.negation_detector {
            *nd = crate::consciousness::negation_detector::NegationDetector::new();
        }
        // Soul has no reset — value alignment is persistent across episodes.
    }
}
