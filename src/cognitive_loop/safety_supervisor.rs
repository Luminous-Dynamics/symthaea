// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Safety Supervisor — Centralized Immune System Orchestration
//!
//! Encapsulates the assessment and enforcement of safety gates within
//! the cognitive loop.

use super::CognitiveLoopService;
use super::safety_enforcement::{self, SafetyEnforcementResult};
use crate::safety::{SafetyAgent, SafetyLevel};

pub struct SafetySupervisor {
    pub agent: SafetyAgent,
    pub guardian_state: crate::cognitive_loop::guardian::GuardianState,
    pub last_result: Option<SafetyEnforcementResult>,
}

impl SafetySupervisor {
    pub fn new() -> Self {
        Self {
            agent: SafetyAgent::new(),
            guardian_state: crate::cognitive_loop::guardian::GuardianState::default(),
            last_result: None,
        }
    }

    /// Compatibility entry point for callers with an explicit numeric PE.
    pub fn assess(
        &mut self,
        consciousness_level: f32,
        prediction_error: f32,
        temporal_coherence: f32,
        integrity_critical: bool,
        cycle: usize,
        #[cfg(feature = "sentinel")] collective_immune: Option<
            &crate::cognitive_loop::collective_immunity::CollectiveImmuneState,
        >,
    ) -> SafetyEnforcementResult {
        self.assess_with_evidence(
            consciousness_level,
            Some(prediction_error),
            temporal_coherence,
            integrity_critical,
            cycle,
            #[cfg(feature = "sentinel")]
            collective_immune,
        )
    }

    /// Assess current state with explicit PE evidence provenance.
    pub fn assess_with_evidence(
        &mut self,
        consciousness_level: f32,
        prediction_error: Option<f32>,
        temporal_coherence: f32,
        integrity_critical: bool,
        cycle: usize,
        #[cfg(feature = "sentinel")] collective_immune: Option<
            &crate::cognitive_loop::collective_immunity::CollectiveImmuneState,
        >,
    ) -> SafetyEnforcementResult {
        let result = safety_enforcement::compute_enforcement_with_evidence(
            &mut self.agent,
            consciousness_level,
            prediction_error,
            temporal_coherence,
            integrity_critical,
            cycle,
            #[cfg(feature = "sentinel")]
            collective_immune,
            #[cfg(not(feature = "sentinel"))]
            None,
        );
        self.last_result = Some(result.clone());
        result
    }
}
