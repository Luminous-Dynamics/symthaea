// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral algebra integration for the cognitive loop.
//!
//! Delegates moral evaluation to the EthicsEngine, which owns the
//! MoralParser and MoralAlgebra instances.

use super::{CognitiveLoopService, MoralJudgmentSummary};

impl CognitiveLoopService {
    /// Evaluate the moral alignment of an input text.
    ///
    /// Delegates to `EthicsEngine::evaluate_moral_input()` (Stage 1) for
    /// HDC-based moral parsing, consent detection, and deontological judgment.
    /// Builds a `MoralJudgmentSummary` from the engine result.
    pub fn evaluate_moral_alignment(&mut self, input: &str) -> MoralJudgmentSummary {
        let result = self.ethics_engine.evaluate_moral_input(input);

        let summary = MoralJudgmentSummary {
            input: input.to_string(),
            verdict: result.verdict,
            deontological_verdict: result.deontological_verdict,
            violations: result.violations,
            satisfactions: result.satisfactions,
            consent_violation: result.consent_violation,
            moral_score: result.moral_score as f32,
            confidence: result.confidence,
        };

        // Store for tracking
        self.ethics_values.last_moral_judgment = Some(summary.clone());
        summary
    }

    /// Get the last moral judgment (if any)
    pub fn last_moral_judgment(&self) -> Option<&MoralJudgmentSummary> {
        self.ethics_values.last_moral_judgment.as_ref()
    }

    /// Check if the last input had moral concerns
    pub fn has_moral_concerns(&self) -> bool {
        self.ethics_values
            .last_moral_judgment
            .as_ref()
            .map(|j| j.moral_score < -0.3 || j.consent_violation || !j.violations.is_empty())
            .unwrap_or(false)
    }
}
