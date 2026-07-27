// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness bridge: bidirectional signal flow between CodingAgent and
//! the cognitive loop's reasoning engine.

use crate::cognitive_loop::{CognitiveLoopService, CycleResult};
#[cfg(feature = "reasoning_engine")]
use crate::consciousness::reasoning_engine::CodeReasoningContext;

#[derive(Debug, Clone, Default)]
pub struct ReasoningFeedback {
    pub gate_blocked: bool,
    /// Whether a tool gate actually ran this cycle (see
    /// `CycleMetadata.reasoning_gate_evaluated`). `gate_blocked` alone cannot
    /// distinguish "gate ran and passed" from "no gate ran" — this can.
    pub gate_evaluated: bool,
    pub reasoning_confidence: f32,
    pub plan_action: Option<usize>,
    pub plan_confidence: f32,
    pub narrative: Option<String>,
}

impl ReasoningFeedback {
    pub fn from_cycle_result(result: &CycleResult) -> Self {
        Self {
            gate_blocked: result.metadata.reasoning_gate_blocked,
            gate_evaluated: result.metadata.reasoning_gate_evaluated,
            reasoning_confidence: result.metadata.reasoning_confidence,
            plan_action: result.metadata.reasoning_plan_action,
            plan_confidence: result.metadata.reasoning_plan_confidence,
            narrative: result.metadata.reasoning_narrative.clone(),
        }
    }
    pub fn should_defer(&self) -> bool {
        self.gate_blocked || self.reasoning_confidence < 0.15
    }
    pub fn should_diagnose(&self) -> bool {
        !self.gate_blocked && self.reasoning_confidence >= 0.15 && self.reasoning_confidence < 0.35
    }
}

/// Captured at the cognitive-loop cycle where a code-generation action was
/// actually decided (Generating/Fixing phase), consumed exactly once when that
/// attempt's outcome resolves (Testing phase). Needed because
/// `record_generation_outcome`'s resolution happens cycles later, after
/// intervening Testing-phase cycles would otherwise overwrite the gate/PE state
/// that belongs to the original decision — see
/// `feedback_plan_review_rigor_standards.md` item 5.
#[derive(Debug, Clone, Copy)]
pub struct PendingReasoningOutcome {
    /// Whether a tool gate actually ran at decision time. If `false`, no posthoc
    /// should be recorded at all — "no gate ran" must never be reported as
    /// "gate passed" (item 4 of the same checklist).
    pub gate_evaluated: bool,
    pub gate_passed: bool,
    pub reasoning_confidence: f32,
    pub prediction_error_at_decision: f64,
}

impl PendingReasoningOutcome {
    pub fn capture(reasoning: &ReasoningFeedback, prediction_error: f64) -> Self {
        Self {
            gate_evaluated: reasoning.gate_evaluated,
            gate_passed: !reasoning.gate_blocked,
            reasoning_confidence: reasoning.reasoning_confidence,
            prediction_error_at_decision: prediction_error,
        }
    }
}

pub struct CodeSignals {
    pub type_confidence: f64,
    pub involves_unsafe: bool,
    pub compile_rate: f64,
    pub retry_count: u32,
    pub has_side_effects: bool,
    pub task_complexity: f64,
    pub syntax_complexity: f32,
    pub algorithm_pattern: f32,
    pub error_likelihood: f32,
}

impl CodeSignals {
    pub fn from_agent_state(
        failure_patterns: &[(String, usize)],
        iteration: usize,
        phase_failures: usize,
        generated_code: Option<&str>,
        energy_budget: f32,
        max_energy: f32,
        native_exhausted: bool,
    ) -> Self {
        let total_failures: usize = failure_patterns.iter().map(|(_, c)| c).sum();
        let compile_rate = if iteration == 0 {
            0.5
        } else {
            1.0 - (total_failures as f64 / (iteration as f64 + 1.0)).min(1.0)
        };
        let involves_unsafe = generated_code
            .map(|c| c.contains("unsafe ") || c.contains("unsafe{"))
            .unwrap_or(false);
        let has_side_effects = generated_code
            .map(|c| {
                c.contains("std::fs::")
                    || c.contains("std::net::")
                    || c.contains("tokio::")
                    || c.contains("File::")
                    || c.contains("Command::")
            })
            .unwrap_or(false);
        let task_complexity = if max_energy > 0.0 {
            (1.0 - energy_budget as f64 / max_energy as f64).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let type_errors = failure_patterns
            .iter()
            .filter(|(p, _)| p.contains("E0308") || p.contains("E0277") || p.contains("E0599"))
            .map(|(_, c)| c)
            .sum::<usize>();
        let type_confidence = (1.0 - (type_errors as f64 / 3.0).min(1.0)).max(0.1);
        let syntax_complexity = generated_code
            .map(|c| {
                let d = c
                    .chars()
                    .fold((0i32, 0i32), |(d, m), ch| match ch {
                        '{' => (d + 1, m.max(d + 1)),
                        '}' => ((d - 1).max(0), m),
                        _ => (d, m),
                    })
                    .1;
                ((d as f32 - 1.0) / 5.0).clamp(0.0, 1.0)
            })
            .unwrap_or(0.0);
        let algorithm_pattern = if native_exhausted { 0.7 } else { 0.3 };
        let error_likelihood = if failure_patterns.is_empty() {
            0.2
        } else {
            (failure_patterns.len() as f32 / 5.0).min(0.9)
        };
        Self {
            type_confidence,
            involves_unsafe,
            compile_rate,
            retry_count: phase_failures as u32,
            has_side_effects,
            task_complexity,
            syntax_complexity,
            algorithm_pattern,
            error_likelihood,
        }
    }
    #[cfg(feature = "reasoning_engine")]
    pub fn to_reasoning_context(&self) -> CodeReasoningContext {
        CodeReasoningContext {
            type_confidence: self.type_confidence,
            involves_unsafe: self.involves_unsafe,
            recent_compile_rate: self.compile_rate,
            retry_count: self.retry_count,
            has_side_effects: self.has_side_effects,
            task_complexity: self.task_complexity,
        }
    }
}

impl CognitiveLoopService {
    #[cfg(feature = "reasoning_engine")]
    pub fn inject_code_context(&mut self, ctx: CodeReasoningContext) {
        self.carryover.injected_code_context = Some(ctx);
    }
    #[cfg(feature = "reasoning_engine")]
    pub fn clear_code_context(&mut self) {
        self.carryover.injected_code_context = None;
    }
    pub fn set_broca_code_channels(&mut self, sc: f32, tc: f32, ap: f32, el: f32) {
        self.language_comm.broca_code_channels = Some([
            sc.clamp(0.0, 1.0),
            tc.clamp(0.0, 1.0),
            ap.clamp(0.0, 1.0),
            el.clamp(0.0, 1.0),
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn code_signals_from_empty_state() {
        let s = CodeSignals::from_agent_state(&[], 0, 0, None, 100.0, 100.0, false);
        assert!((s.compile_rate - 0.5).abs() < f64::EPSILON);
        assert!(!s.involves_unsafe);
    }
    #[test]
    fn code_signals_detects_unsafe() {
        let s = CodeSignals::from_agent_state(
            &[],
            1,
            0,
            Some("fn f() { unsafe { *p } }"),
            100.0,
            100.0,
            false,
        );
        assert!(s.involves_unsafe);
    }
    #[test]
    fn code_signals_detects_side_effects() {
        let s = CodeSignals::from_agent_state(
            &[],
            1,
            0,
            Some("use std::fs::File;"),
            100.0,
            100.0,
            false,
        );
        assert!(s.has_side_effects);
    }
    #[test]
    fn code_signals_compile_rate_degrades() {
        let s = CodeSignals::from_agent_state(
            &[("E0308".into(), 3), ("E0425".into(), 2)],
            10,
            2,
            None,
            50.0,
            100.0,
            false,
        );
        assert!(s.compile_rate < 0.6 && s.compile_rate > 0.4);
    }
    #[test]
    fn code_signals_type_confidence_drops() {
        let s = CodeSignals::from_agent_state(
            &[("E0308: mismatch".into(), 3)],
            5,
            1,
            None,
            100.0,
            100.0,
            false,
        );
        assert!((s.type_confidence - 0.1).abs() < f64::EPSILON);
    }
    #[test]
    fn code_signals_task_complexity() {
        let s = CodeSignals::from_agent_state(&[], 5, 0, None, 25.0, 100.0, false);
        assert!((s.task_complexity - 0.75).abs() < 0.01);
    }
    #[test]
    fn reasoning_defer_gate() {
        let f = ReasoningFeedback {
            gate_blocked: true,
            reasoning_confidence: 0.8,
            ..Default::default()
        };
        assert!(f.should_defer());
    }
    #[test]
    fn reasoning_defer_low() {
        let f = ReasoningFeedback {
            reasoning_confidence: 0.1,
            ..Default::default()
        };
        assert!(f.should_defer());
    }
    #[test]
    fn reasoning_diagnose_mid() {
        let f = ReasoningFeedback {
            reasoning_confidence: 0.25,
            ..Default::default()
        };
        assert!(f.should_diagnose());
        assert!(!f.should_defer());
    }
    #[test]
    fn reasoning_normal() {
        let f = ReasoningFeedback {
            reasoning_confidence: 0.7,
            ..Default::default()
        };
        assert!(!f.should_defer());
        assert!(!f.should_diagnose());
    }
    #[cfg(feature = "reasoning_engine")]
    #[test]
    fn roundtrip() {
        let s = CodeSignals::from_agent_state(
            &[("E0308".into(), 1)],
            5,
            2,
            Some("unsafe { }"),
            50.0,
            100.0,
            true,
        );
        let c = s.to_reasoning_context();
        assert!(c.involves_unsafe);
        assert_eq!(c.retry_count, 2);
    }
}
