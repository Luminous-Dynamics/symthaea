//! Moral algebra integration for the cognitive loop.
//!
//! Evaluates ethical alignment of inputs using HDC-based moral algebra,
//! consent detection, and deontological judgment.

use crate::hdc::moral_algebra::{DeontologicalVerdict, MoralVerdict};

use super::{CognitiveLoopService, MoralJudgmentSummary};

impl CognitiveLoopService {
    /// Evaluate the moral alignment of an input text.
    ///
    /// Uses HDC-based moral algebra to:
    /// - Extract moral primitives (agent, patient, action, intent, consent, magnitude)
    /// - Check for consent violations
    /// - Check for deontological violations/satisfactions
    /// - Compute overall moral score
    ///
    /// Returns a summary of the moral evaluation.
    pub fn evaluate_moral_alignment(&mut self, input: &str) -> MoralJudgmentSummary {
        // Parse and encode the input
        let encoded = self
            .moral_parser
            .parse_and_encode(input, &self.moral_algebra);

        // Get basic judgment
        let (verdict_str, good_sim, bad_sim) =
            if let Some(judgment) = encoded.judge(&self.moral_algebra) {
                let v = match judgment.verdict {
                    MoralVerdict::Good => "Good",
                    MoralVerdict::Bad => "Bad",
                    MoralVerdict::Neutral => "Neutral",
                    MoralVerdict::ConsentViolation => "ConsentViolation",
                };
                (
                    v.to_string(),
                    judgment.good_similarity,
                    judgment.bad_similarity,
                )
            } else {
                ("Neutral".to_string(), 0.0, 0.0)
            };

        // Get deontological judgment
        let deont = self.moral_algebra.judge_deontological(input);
        let deont_verdict_str = match deont.verdict {
            DeontologicalVerdict::RightDutyFulfilled => "Permissible",
            DeontologicalVerdict::WrongPerfectDutyViolated => "Impermissible",
            DeontologicalVerdict::WrongImperfectDutyViolated => "Impermissible",
            DeontologicalVerdict::Neutral => "Neutral",
        }
        .to_string();

        // Extract violation and satisfaction names
        let violations: Vec<String> = deont
            .violations
            .iter()
            .map(|v| v.rule_name.clone())
            .collect();
        let satisfactions: Vec<String> = deont
            .satisfactions
            .iter()
            .map(|s| s.rule_name.clone())
            .collect();

        // Check consent violation
        let consent_violation = encoded.is_consent_violation();

        // Compute moral score (-1.0 to 1.0)
        let moral_score = if consent_violation {
            -0.8 // Strong penalty for consent violation
        } else {
            // Balance good/bad similarity and deontological score
            let base_score = (good_sim - bad_sim).clamp(-1.0, 1.0);
            let deont_factor = deont.score.clamp(-1.0, 1.0);
            (base_score * 0.6 + deont_factor * 0.4).clamp(-1.0, 1.0)
        };

        // Compute confidence based on parsing quality
        let confidence = encoded.parsed.confidence;

        let summary = MoralJudgmentSummary {
            input: input.to_string(),
            verdict: verdict_str,
            deontological_verdict: deont_verdict_str,
            violations,
            satisfactions,
            consent_violation,
            moral_score,
            confidence,
        };

        // Store for tracking
        self.last_moral_judgment = Some(summary.clone());
        summary
    }

    /// Get the last moral judgment (if any)
    pub fn last_moral_judgment(&self) -> Option<&MoralJudgmentSummary> {
        self.last_moral_judgment.as_ref()
    }

    /// Check if the last input had moral concerns
    pub fn has_moral_concerns(&self) -> bool {
        self.last_moral_judgment
            .as_ref()
            .map(|j| j.moral_score < -0.3 || j.consent_violation || !j.violations.is_empty())
            .unwrap_or(false)
    }
}
