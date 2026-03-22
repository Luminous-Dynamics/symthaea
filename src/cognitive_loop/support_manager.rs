// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Support Intelligence Manager — consolidates support subsystem fields.
//!
//! Groups predictive diagnostics, knowledge federation, triage, privacy,
//! and action engine into a single manager behind `#[cfg(feature = "support")]`.

/// Consolidated support intelligence subsystem.
///
/// Holds all support-feature-gated fields that were previously scattered
/// across `CognitiveLoopService`. Only used in `cycle_phase_feedback.rs`
/// Phase 10 (Support Intelligence).
#[derive(Debug)]
pub struct SupportManager {
    /// Predictive engine for zero-click proactive support (telemetry → free energy alerts).
    pub predictive_engine: Option<symthaea_support::predictive::PredictiveEngine>,

    /// Knowledge manager for article graduation and cognitive update absorption.
    pub knowledge_manager: Option<symthaea_support::knowledge::KnowledgeManager>,

    /// Triage engine for ticket classification and prioritization.
    pub triage_engine: Option<symthaea_support::triage::TriageEngine>,

    /// Privacy manager for federation sharing tier enforcement.
    pub privacy_manager: Option<symthaea_support::privacy::PrivacyManager>,

    /// Action engine for autonomous remediation proposals (reserved).
    #[allow(dead_code)]
    pub action_engine: Option<symthaea_support::actions::ActionEngine>,

    /// Cycle counter for amortizing support subsystem updates.
    pub cycle_counter: u64,
}

impl SupportManager {
    /// Create a new fully-initialized SupportManager.
    pub fn new() -> Self {
        Self {
            predictive_engine: Some(symthaea_support::predictive::PredictiveEngine::new()),
            knowledge_manager: Some(symthaea_support::knowledge::KnowledgeManager::new()),
            triage_engine: Some(symthaea_support::triage::TriageEngine::new()),
            privacy_manager: Some(symthaea_support::privacy::PrivacyManager::default()),
            action_engine: Some(symthaea_support::actions::ActionEngine::new()),
            cycle_counter: 0,
        }
    }

    /// Reset support state (counter only — engines are stateless).
    pub fn reset(&mut self) {
        self.cycle_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support_manager_new() {
        let mgr = SupportManager::new();
        assert!(mgr.predictive_engine.is_some());
        assert!(mgr.knowledge_manager.is_some());
        assert!(mgr.triage_engine.is_some());
        assert!(mgr.privacy_manager.is_some());
        assert!(mgr.action_engine.is_some());
        assert_eq!(mgr.cycle_counter, 0);
    }

    #[test]
    fn test_support_manager_reset() {
        let mut mgr = SupportManager::new();
        mgr.cycle_counter = 100;
        mgr.reset();
        assert_eq!(mgr.cycle_counter, 0);
    }
}
