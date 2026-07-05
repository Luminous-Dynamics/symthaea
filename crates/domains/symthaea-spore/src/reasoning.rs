// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Simplified Reasoning Engine for the Spore WASM kernel.
//!
//! Ports the desktop ConsciousReasoningEngine's 7-step cycle into a
//! lightweight version suitable for wasm32-unknown-unknown targets.
//!
//! The 7 steps mirror the desktop pipeline (mod.rs):
//! 1. Perceive  — encode input to HDC
//! 2. Assess    — evaluate consciousness state + reliability
//! 3. Plan      — generate candidate actions (simplified MCTS)
//! 4. Predict   — estimate outcomes via CfC forward projection
//! 5. Evaluate  — score candidates (expected value × confidence × ethics)
//! 6. Select    — pick best action
//! 7. Reflect   — meta-cognitive review (Phi delta, prediction error)
//!
//! Budget: the entire cycle must complete in <5ms on a mobile browser.

use serde::{Deserialize, Serialize};

/// The 7 reasoning steps, matching the desktop engine's cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningStep {
    Perceive,
    Assess,
    Plan,
    Predict,
    Evaluate,
    Select,
    Reflect,
}

impl ReasoningStep {
    /// All steps in order.
    pub const ALL: [ReasoningStep; 7] = [
        Self::Perceive,
        Self::Assess,
        Self::Plan,
        Self::Predict,
        Self::Evaluate,
        Self::Select,
        Self::Reflect,
    ];

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Perceive => "Perceive",
            Self::Assess => "Assess",
            Self::Plan => "Plan",
            Self::Predict => "Predict",
            Self::Evaluate => "Evaluate",
            Self::Select => "Select",
            Self::Reflect => "Reflect",
        }
    }
}

/// Result of a single reasoning step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStepResult {
    pub step: ReasoningStep,
    /// Change in Phi during this step.
    pub phi_change: f32,
    /// Prediction error observed.
    pub prediction_error: f32,
    /// Brief description of what happened.
    pub description: String,
}

/// A candidate action generated during the Plan step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCandidate {
    /// What this action would do.
    pub description: String,
    /// Expected value (reward estimate, 0.0-1.0).
    pub expected_value: f32,
    /// Confidence in the estimate (0.0-1.0).
    pub confidence: f32,
    /// Ethical alignment score (0.0-1.0).
    pub ethical_score: f32,
}

impl ActionCandidate {
    /// Combined score: EV × confidence × ethics.
    pub fn combined_score(&self) -> f32 {
        self.expected_value * self.confidence * self.ethical_score
    }
}

/// The full result of a 7-step reasoning cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningCycle {
    /// Which step we're currently on (Complete after run).
    pub current_step: ReasoningStep,
    /// Results from each completed step.
    pub steps_completed: Vec<ReasoningStepResult>,
    /// Candidate actions generated in the Plan step.
    pub candidates: Vec<ActionCandidate>,
    /// The action selected in the Select step (if any).
    pub selected_action: Option<ActionCandidate>,
    /// Consciousness level at start.
    pub initial_consciousness: f32,
    /// Consciousness level at end (after Reflect).
    pub final_consciousness: f32,
    /// Total reasoning time (simulated, not wall-clock in WASM).
    pub total_time_ms: f32,
}

/// Lightweight reasoning engine for the Spore.
pub struct ReasoningEngine {
    /// Running cycle count.
    cycle_count: u64,
    /// Last prediction error for delta tracking.
    last_prediction_error: f32,
    /// Last consciousness for delta tracking.
    last_consciousness: f32,
}

impl Default for ReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningEngine {
    pub fn new() -> Self {
        Self {
            cycle_count: 0,
            last_prediction_error: 0.0,
            last_consciousness: 0.0,
        }
    }

    /// Run the full 7-step reasoning cycle.
    ///
    /// # Arguments
    /// * `input` - text input being reasoned about
    /// * `consciousness` - current consciousness level (from Phi equation)
    /// * `prediction_error` - current PE from CfC network
    /// * `neuromods` - neuromodulator levels [DA, NE, 5-HT, OT, ...]
    /// * `harmony_alignment` - ethical alignment score
    pub fn run(
        &mut self,
        input: &str,
        consciousness: f32,
        prediction_error: f32,
        neuromods: &[f32],
        harmony_alignment: f32,
    ) -> ReasoningCycle {
        self.cycle_count += 1;
        let mut steps = Vec::with_capacity(7);

        let da = neuromods.first().copied().unwrap_or(0.5);
        let ne = neuromods.get(1).copied().unwrap_or(0.5);
        let serotonin = neuromods.get(2).copied().unwrap_or(0.5);

        // ── STEP 1: PERCEIVE ─────────────────────────────────────────
        // In the desktop, this encodes input via HDC. Here we note input
        // characteristics that affect downstream reasoning.
        let input_complexity = (input.len() as f32 / 100.0).min(1.0);
        let phi_base = consciousness;
        steps.push(ReasoningStepResult {
            step: ReasoningStep::Perceive,
            phi_change: 0.0,
            prediction_error,
            description: format!(
                "Encoded input ({} chars, complexity {:.2})",
                input.len(),
                input_complexity
            ),
        });

        // ── STEP 2: ASSESS ───────────────────────────────────────────
        // Desktop computes reliability R and effective Phi = Phi × R^gamma.
        // Simplified: reliability from prediction error stability.
        let pe_delta = (prediction_error - self.last_prediction_error).abs();
        let reliability = (1.0 - pe_delta).clamp(0.1, 1.0);
        let effective_phi = consciousness * reliability;
        let phi_change_assess = effective_phi - phi_base;
        steps.push(ReasoningStepResult {
            step: ReasoningStep::Assess,
            phi_change: phi_change_assess,
            prediction_error: pe_delta,
            description: format!(
                "Reliability={:.3}, Phi_eff={:.3} (raw={:.3})",
                reliability, effective_phi, consciousness
            ),
        });

        // ── STEP 3: PLAN ─────────────────────────────────────────────
        // Desktop uses MCTS with O(1) rollouts. We generate 3-5 candidates
        // using heuristics seeded by neuromodulator state.
        let mut candidates = Vec::with_capacity(4);

        // Candidate 1: Exploitative (high DA → exploit known patterns)
        candidates.push(ActionCandidate {
            description: "Exploit: apply learned pattern".into(),
            expected_value: da * 0.8 + serotonin * 0.2,
            confidence: reliability * 0.9,
            ethical_score: harmony_alignment,
        });

        // Candidate 2: Explorative (high NE → explore new territory)
        candidates.push(ActionCandidate {
            description: "Explore: seek novel connections".into(),
            expected_value: ne * 0.7 + prediction_error * 0.3,
            confidence: reliability * 0.6,
            ethical_score: harmony_alignment,
        });

        // Candidate 3: Reflective (high 5-HT → consolidate understanding)
        candidates.push(ActionCandidate {
            description: "Reflect: consolidate and integrate".into(),
            expected_value: serotonin * 0.6 + consciousness * 0.4,
            confidence: reliability * 0.8,
            ethical_score: harmony_alignment * 1.05_f32.min(1.0),
        });

        // Candidate 4: Creative (high PE + moderate consciousness)
        if prediction_error > 0.3 && consciousness > 0.2 {
            candidates.push(ActionCandidate {
                description: "Create: synthesize novel interpretation".into(),
                expected_value: prediction_error * 0.5 + consciousness * 0.5,
                confidence: reliability * 0.5,
                ethical_score: harmony_alignment,
            });
        }

        steps.push(ReasoningStepResult {
            step: ReasoningStep::Plan,
            phi_change: 0.0,
            prediction_error,
            description: format!("Generated {} candidate actions", candidates.len()),
        });

        // ── STEP 4: PREDICT ──────────────────────────────────────────
        // Desktop does CfC forward projection. We estimate outcome variance.
        let outcome_variance: f32 = candidates
            .iter()
            .map(|c| (c.expected_value - 0.5).powi(2))
            .sum::<f32>()
            / candidates.len().max(1) as f32;
        steps.push(ReasoningStepResult {
            step: ReasoningStep::Predict,
            phi_change: 0.0,
            prediction_error: outcome_variance,
            description: format!("Outcome variance: {:.4}", outcome_variance),
        });

        // ── STEP 5: EVALUATE ─────────────────────────────────────────
        // Score all candidates by combined metric.
        for c in &mut candidates {
            // Clamp ethical score
            c.ethical_score = c.ethical_score.clamp(0.0, 1.0);
        }
        candidates.sort_by(|a, b| {
            b.combined_score()
                .partial_cmp(&a.combined_score())
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        steps.push(ReasoningStepResult {
            step: ReasoningStep::Evaluate,
            phi_change: 0.0,
            prediction_error,
            description: format!(
                "Best score: {:.3}",
                candidates.first().map_or(0.0, |c| c.combined_score())
            ),
        });

        // ── STEP 6: SELECT ───────────────────────────────────────────
        // Gate by effective Phi: need Phi_eff > 0.1 to act.
        let selected = if effective_phi > 0.1 {
            candidates.first().cloned()
        } else {
            None
        };
        let select_desc = match &selected {
            Some(a) => format!(
                "Selected: {} (score={:.3})",
                a.description,
                a.combined_score()
            ),
            None => format!("No action (Phi_eff={:.3} < 0.1 threshold)", effective_phi),
        };
        steps.push(ReasoningStepResult {
            step: ReasoningStep::Select,
            phi_change: 0.0,
            prediction_error,
            description: select_desc,
        });

        // ── STEP 7: REFLECT ──────────────────────────────────────────
        // Meta-cognitive review: how did this cycle compare to the last?
        let consciousness_delta = consciousness - self.last_consciousness;
        let reflect_phi_change = consciousness_delta * 0.1; // reflection slightly modulates Phi
        steps.push(ReasoningStepResult {
            step: ReasoningStep::Reflect,
            phi_change: reflect_phi_change,
            prediction_error: pe_delta,
            description: format!(
                "Consciousness delta: {:.4}, cycle {}",
                consciousness_delta, self.cycle_count
            ),
        });

        // Update tracking state
        self.last_prediction_error = prediction_error;
        self.last_consciousness = consciousness;

        let total_phi_change: f32 = steps.iter().map(|s| s.phi_change).sum();
        let final_consciousness = (consciousness + total_phi_change).clamp(0.0, 1.0);

        ReasoningCycle {
            current_step: ReasoningStep::Reflect,
            steps_completed: steps,
            candidates,
            selected_action: selected,
            initial_consciousness: consciousness,
            final_consciousness,
            total_time_ms: 0.0, // WASM doesn't have reliable timing
        }
    }

    /// Current cycle count.
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_cycle_completes_7_steps() {
        let mut engine = ReasoningEngine::new();
        let result = engine.run("test input", 0.5, 0.3, &[0.6, 0.5, 0.4, 0.3], 0.8);
        assert_eq!(result.steps_completed.len(), 7);
        assert_eq!(result.current_step, ReasoningStep::Reflect);
        assert!(!result.candidates.is_empty());
    }

    #[test]
    fn test_low_phi_gates_action_selection() {
        let mut engine = ReasoningEngine::new();
        // Very low consciousness should prevent action selection
        let result = engine.run("test", 0.05, 0.1, &[0.5, 0.5, 0.5, 0.5], 0.9);
        assert!(
            result.selected_action.is_none(),
            "Should not select action with Phi_eff < 0.1"
        );
    }

    #[test]
    fn test_ethical_score_clamps() {
        let mut engine = ReasoningEngine::new();
        let result = engine.run("test", 0.8, 0.2, &[0.9, 0.9, 0.9, 0.9], 1.0);
        for c in &result.candidates {
            assert!(c.ethical_score <= 1.0);
            assert!(c.ethical_score >= 0.0);
        }
    }

    #[test]
    fn test_creative_candidate_requires_high_pe() {
        let mut engine = ReasoningEngine::new();
        // Low PE: should have 3 candidates
        let result_low = engine.run("test", 0.5, 0.1, &[0.5, 0.5, 0.5, 0.5], 0.8);
        // High PE: should have 4 candidates (creative added)
        let result_high = engine.run("test", 0.5, 0.5, &[0.5, 0.5, 0.5, 0.5], 0.8);
        assert!(result_high.candidates.len() > result_low.candidates.len());
    }
}
