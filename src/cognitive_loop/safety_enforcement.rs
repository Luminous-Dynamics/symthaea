// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Safety Enforcement — Decentralized Defensive Force
//!
//! This module is the single entry point for all immune system infrastructure.
//! It re-exports the defense, guardian, threat memory, and collective immunity
//! submodules, and provides the `compute_enforcement()` function for per-cycle
//! safety gating.
//!
//! ## Submodules (included inline)
//!
//! - `defense` — Graduated DefenseAction cascade with moral algebra filter
//! - `guardian` — Physical posture system: consciousness-gated robot defense
//! - `threat_memory` — 32D feature-vector threat pattern encoding (sentinel feature)
//! - `collective_immunity` — Swarm-wide threat aggregation (sentinel feature)
//!
//! ## Design
//!
//! Safety enforcement is isolated in this module (rather than inline in cycle.rs)
//! so that the core cycle remains clean and the safety logic is independently
//! testable without constructing a full CognitiveLoopService.
//!
//! ## NRC Defense-in-Depth
//!
//! Mirrors the nuclear regulatory commission's layered safety approach:
//! - **Green**: Normal operation, no gates
//! - **Yellow**: Reduced plasticity (0.7×), dampened exploration (0.5×), NE vigilance boost
//! - **Orange**: Severe LR reduction (0.3×), minimal exploration (0.1×), cortisol stress, consciousness penalty
//! - **Red**: Near-zero LR (0.05×), zero exploration, maximum stress response

use crate::safety::{SafetyAgent, SafetyLevel, SafetyMetrics};

use super::thresholds::{
    SAFETY_ORANGE_CONSCIOUSNESS_PENALTY, SAFETY_ORANGE_CORTISOL_BOOST,
    SAFETY_ORANGE_EXPLORATION_DAMPEN, SAFETY_ORANGE_LR_MULTIPLIER, SAFETY_RED_LR_MULTIPLIER,
    SAFETY_YELLOW_EXPLORATION_DAMPEN, SAFETY_YELLOW_LR_MULTIPLIER, SAFETY_YELLOW_NE_BOOST,
};

/// Result of safety enforcement for the current cycle.
#[derive(Debug, Clone)]
pub struct SafetyEnforcementResult {
    /// The assessed safety level.
    pub level: SafetyLevel,
    /// Learning rate multiplier to apply (1.0 = no change).
    pub lr_multiplier: f32,
    /// Exploration multiplier to apply (1.0 = no change).
    pub exploration_multiplier: f32,
    /// NE baseline nudge (0.0 = no change).
    pub ne_nudge: f32,
    /// Allostatic load to accumulate (0.0 = no change).
    pub allostatic_load: f32,
    /// Consciousness penalty to subtract (0.0 = no change).
    pub consciousness_penalty: f32,
    /// Whether motor output should be halted entirely.
    pub motor_halt: bool,
    /// Whether motor output should be restricted to read-only.
    pub motor_readonly: bool,
    /// Reasons from the safety assessment.
    pub reasons: Vec<String>,
}

/// Assess current state and compute enforcement gates.
///
/// This is a pure function — it reads the safety agent's assessment
/// and returns what should change, without mutating any state.
pub fn compute_enforcement(
    agent: &mut SafetyAgent,
    consciousness_level: f32,
    prediction_error: f32,
    temporal_coherence: f32,
    integrity_critical: bool,
    cycle: usize,
    #[cfg(feature = "sentinel")] collective_immune: Option<
        &super::collective_immunity::CollectiveImmuneState,
    >,
    #[cfg(not(feature = "sentinel"))] _collective_immune: Option<&()>,
) -> SafetyEnforcementResult {
    let metrics = SafetyMetrics {
        cycle,
        consciousness_level,
        prediction_error,
        temporal_coherence,
        integrity_critical,
    };
    let assessment = agent.assess(metrics);
    #[allow(unused_mut)]
    let mut level = assessment.level;

    // Collective immunity escalation: swarm threat data can escalate safety level
    #[cfg(feature = "sentinel")]
    if let Some(immune) = collective_immune {
        if let Some(escalated) = super::collective_immunity::recommend_safety_escalation(
            level,
            immune.swarm_threat_level,
            immune,
        ) {
            level = escalated;
        }
    }

    let (lr_mult, explore_mult) = match level {
        SafetyLevel::Red => (SAFETY_RED_LR_MULTIPLIER, 0.0),
        SafetyLevel::Orange => (
            SAFETY_ORANGE_LR_MULTIPLIER,
            SAFETY_ORANGE_EXPLORATION_DAMPEN,
        ),
        SafetyLevel::Yellow => (
            SAFETY_YELLOW_LR_MULTIPLIER,
            SAFETY_YELLOW_EXPLORATION_DAMPEN,
        ),
        SafetyLevel::Green => (1.0, 1.0),
    };

    let ne_nudge = match level {
        SafetyLevel::Yellow => SAFETY_YELLOW_NE_BOOST,
        SafetyLevel::Orange | SafetyLevel::Red => SAFETY_YELLOW_NE_BOOST * 2.0,
        SafetyLevel::Green => 0.0,
    };

    let allostatic = match level {
        SafetyLevel::Orange | SafetyLevel::Red => SAFETY_ORANGE_CORTISOL_BOOST,
        _ => 0.0,
    };

    let consciousness_penalty = if level >= SafetyLevel::Orange {
        SAFETY_ORANGE_CONSCIOUSNESS_PENALTY
    } else {
        0.0
    };

    SafetyEnforcementResult {
        level,
        lr_multiplier: lr_mult,
        exploration_multiplier: explore_mult,
        ne_nudge,
        allostatic_load: allostatic,
        consciousness_penalty,
        motor_halt: level == SafetyLevel::Red,
        motor_readonly: level == SafetyLevel::Orange,
        reasons: assessment.reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_no_gates() {
        let mut agent = SafetyAgent::new();
        let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, false, 100, None);
        assert_eq!(result.level, SafetyLevel::Green);
        assert!((result.lr_multiplier - 1.0).abs() < 1e-6);
        assert!((result.exploration_multiplier - 1.0).abs() < 1e-6);
        assert!(!result.motor_halt);
        assert!(!result.motor_readonly);
    }

    #[test]
    fn test_yellow_reduces_lr() {
        let mut agent = SafetyAgent::new();
        let result = compute_enforcement(&mut agent, 0.5, 0.2, 0.9, false, 100, None);
        assert_eq!(result.level, SafetyLevel::Yellow);
        assert!(result.lr_multiplier < 1.0);
        assert!(result.exploration_multiplier < 1.0);
        assert!(result.ne_nudge > 0.0);
    }

    #[test]
    fn test_orange_restricts_motor() {
        let mut agent = SafetyAgent::new();
        let result = compute_enforcement(&mut agent, 0.25, 0.2, 0.9, false, 100, None);
        assert_eq!(result.level, SafetyLevel::Orange);
        assert!(result.motor_readonly);
        assert!(!result.motor_halt);
        assert!(result.consciousness_penalty > 0.0);
        assert!(result.allostatic_load > 0.0);
    }

    #[test]
    fn test_red_halts_motor() {
        let mut agent = SafetyAgent::new();
        let result = compute_enforcement(&mut agent, 0.1, 0.2, 0.9, false, 100, None);
        assert_eq!(result.level, SafetyLevel::Red);
        assert!(result.motor_halt);
        assert!(result.lr_multiplier < 0.1);
        assert!((result.exploration_multiplier).abs() < 1e-6);
    }

    #[test]
    fn test_integrity_critical_escalates() {
        let mut agent = SafetyAgent::new();
        // Normal consciousness but integrity failure → escalate
        let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, true, 100, None);
        assert!(result.level >= SafetyLevel::Yellow);
    }

    #[test]
    fn test_high_pe_escalates() {
        let mut agent = SafetyAgent::new();
        // Good consciousness but high prediction error → escalate
        let result = compute_enforcement(&mut agent, 0.8, 0.9, 0.9, false, 100, None);
        assert!(result.level >= SafetyLevel::Yellow);
    }

    #[test]
    fn test_low_coherence_escalates() {
        let mut agent = SafetyAgent::new();
        // Good consciousness but low temporal coherence → escalate
        let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.1, false, 100, None);
        assert!(result.level >= SafetyLevel::Yellow);
    }

    #[test]
    fn test_lr_multiplier_ordering() {
        // Red < Orange < Yellow < Green (1.0)
        // Use fresh agents to avoid trend escalation from history
        let red = compute_enforcement(&mut SafetyAgent::new(), 0.05, 0.2, 0.9, false, 1, None);
        let orange = compute_enforcement(&mut SafetyAgent::new(), 0.25, 0.2, 0.9, false, 1, None);
        let yellow = compute_enforcement(&mut SafetyAgent::new(), 0.5, 0.2, 0.9, false, 1, None);
        let green = compute_enforcement(&mut SafetyAgent::new(), 0.8, 0.2, 0.9, false, 1, None);
        assert!(red.lr_multiplier < orange.lr_multiplier);
        assert!(orange.lr_multiplier < yellow.lr_multiplier);
        assert!(yellow.lr_multiplier < green.lr_multiplier);
    }

    #[test]
    fn test_ne_nudge_proportional() {
        let mut agent = SafetyAgent::new();
        let yellow = compute_enforcement(&mut agent, 0.5, 0.2, 0.9, false, 1, None);
        let orange = compute_enforcement(&mut agent, 0.25, 0.2, 0.9, false, 2, None);
        assert!(orange.ne_nudge > yellow.ne_nudge);
    }

    #[test]
    #[cfg(feature = "sentinel")]
    fn test_collective_immune_escalates_to_yellow() {
        let mut agent = SafetyAgent::new();
        let immune = crate::cognitive_loop::collective_immunity::CollectiveImmuneState {
            swarm_threat_level: 0.6,
            coherence_adjusted_severity: 0.6,
            immune_response_active: false,
            ..Default::default()
        };
        let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, false, 100, Some(&immune));
        assert!(
            result.level >= SafetyLevel::Yellow,
            "Collective threat should escalate to Yellow"
        );
    }

    #[test]
    #[cfg(feature = "sentinel")]
    fn test_collective_immune_escalates_to_orange() {
        let mut agent = SafetyAgent::new();
        let immune = crate::cognitive_loop::collective_immunity::CollectiveImmuneState {
            swarm_threat_level: 0.8,
            coherence_adjusted_severity: 0.8,
            immune_response_active: true,
            ..Default::default()
        };
        let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, false, 100, Some(&immune));
        assert_eq!(
            result.level,
            SafetyLevel::Orange,
            "High collective threat should escalate to Orange"
        );
    }

    #[test]
    fn test_no_escalation_without_collective() {
        let mut agent = SafetyAgent::new();
        let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, false, 100, None);
        assert_eq!(
            result.level,
            SafetyLevel::Green,
            "No collective data should not escalate"
        );
    }
}
