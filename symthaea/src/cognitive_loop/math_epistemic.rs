#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Epistemic Gating for Mathematical Claims (Phase 7b)
//!
//! Applies epistemic constraints to math outputs so that Symthaea
//! knows WHAT IT DOESN'T KNOW about its mathematical answers.
//!
//! Every math result carries an `EpistemicMathResult` with:
//! - Numerical error bounds
//! - Multi-method agreement fraction
//! - Soundness classification (Verified / Probable / Uncertain)
//! - Explicit caveats ("what I don't know")
//!
//! Science: Lakatos (1976) — mathematical knowledge is fallible and
//! progresses through proofs and refutations. Polya (1954) — plausible
//! reasoning in mathematics requires tracking confidence.

use serde::{Deserialize, Serialize};

// ─── Soundness Level ──────────────────────────────────────────────────────────

/// Classification of mathematical claim soundness.
///
/// Ordered from strongest to weakest evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SoundnessLevel {
    /// Multi-path verified, error < 1e-10.
    /// Multiple independent methods agree on the result.
    Verified,
    /// Single method, error < 1e-6.
    /// Result is likely correct but lacks independent verification.
    Probable,
    /// No verification, or large error.
    /// Result should be treated with caution.
    Uncertain,
}

impl SoundnessLevel {
    /// Return a human-readable label.
    pub fn as_str(&self) -> &'static str {
        match self {
            SoundnessLevel::Verified => "Verified",
            SoundnessLevel::Probable => "Probable",
            SoundnessLevel::Uncertain => "Uncertain",
        }
    }

    /// Numeric confidence floor for this soundness level.
    pub fn confidence_floor(&self) -> f64 {
        match self {
            SoundnessLevel::Verified => 0.95,
            SoundnessLevel::Probable => 0.70,
            SoundnessLevel::Uncertain => 0.30,
        }
    }
}

impl Default for SoundnessLevel {
    fn default() -> Self {
        SoundnessLevel::Uncertain
    }
}

// ─── Epistemic Math Result ────────────────────────────────────────────────────

/// A math result with full epistemic metadata.
///
/// Wraps a numerical answer with everything the system knows (and
/// doesn't know) about the quality of that answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicMathResult {
    /// The computed value.
    pub value: f64,
    /// Estimated numerical error bound (e.g., floating-point, truncation).
    pub error_bound: f64,
    /// Overall confidence in the result (0.0-1.0).
    pub confidence: f64,
    /// Fraction of methods that agree on the result (0.0-1.0).
    /// 1.0 means all tried methods agree; 0.0 means a single method was used.
    pub method_agreement: f64,
    /// Explicit caveats: things the system doesn't know or can't guarantee.
    pub caveats: Vec<String>,
    /// Soundness classification derived from evidence quality.
    pub soundness: SoundnessLevel,
}

impl EpistemicMathResult {
    /// Determine soundness level from error bound, method agreement, and
    /// whether multi-path verification was performed.
    pub fn classify_soundness(
        error_bound: f64,
        method_agreement: f64,
        multipath_verified: bool,
    ) -> SoundnessLevel {
        if multipath_verified && error_bound < 1e-10 && method_agreement >= 0.9 {
            SoundnessLevel::Verified
        } else if error_bound < 1e-6 && method_agreement >= 0.5 {
            SoundnessLevel::Probable
        } else {
            SoundnessLevel::Uncertain
        }
    }

    /// Create an epistemic result from a raw computation.
    ///
    /// # Arguments
    /// * `value` — the computed numerical result
    /// * `error_bound` — estimated error (0.0 if exact)
    /// * `multipath_verified` — whether multiple methods agree
    /// * `method_count` — total methods tried
    /// * `agreeing_methods` — methods that agree within tolerance
    /// * `base_confidence` — solver's own confidence estimate
    pub fn from_computation(
        value: f64,
        error_bound: f64,
        multipath_verified: bool,
        method_count: usize,
        agreeing_methods: usize,
        base_confidence: f64,
    ) -> Self {
        let method_agreement = if method_count > 0 {
            agreeing_methods as f64 / method_count as f64
        } else {
            0.0
        };

        let soundness = Self::classify_soundness(error_bound, method_agreement, multipath_verified);

        // Confidence is the max of base_confidence and soundness floor,
        // scaled by method agreement.
        let confidence = base_confidence.max(soundness.confidence_floor()).min(1.0)
            * (0.5 + 0.5 * method_agreement);

        let mut caveats = Vec::new();

        // Generate caveats based on what we don't know
        if !multipath_verified {
            caveats.push("Result not independently verified by multiple methods".to_string());
        }
        if error_bound > 1e-6 {
            caveats.push(format!(
                "Numerical error may be significant (bound: {:.2e})",
                error_bound
            ));
        }
        if method_agreement < 0.5 && method_count > 1 {
            caveats.push(format!(
                "Low method agreement ({}/{} methods agree)",
                agreeing_methods, method_count
            ));
        }
        if value.is_nan() {
            caveats.push("Result is NaN; computation may be ill-defined".to_string());
        }
        if value.is_infinite() {
            caveats.push("Result is infinite; possible singularity or overflow".to_string());
        }
        if method_count == 1 {
            caveats.push("Only one method available; no cross-validation possible".to_string());
        }

        EpistemicMathResult {
            value,
            error_bound,
            confidence,
            method_agreement,
            caveats,
            soundness,
        }
    }

    /// Create a verified result (all checks pass, high confidence).
    pub fn verified(value: f64, error_bound: f64) -> Self {
        Self::from_computation(value, error_bound, true, 3, 3, 0.99)
    }

    /// Create a probable result (single method, reasonable confidence).
    pub fn probable(value: f64, error_bound: f64) -> Self {
        Self::from_computation(value, error_bound, false, 1, 1, 0.80)
    }

    /// Create an uncertain result (no verification, low confidence).
    pub fn uncertain(value: f64, error_bound: f64) -> Self {
        Self::from_computation(value, error_bound, false, 1, 0, 0.30)
    }

    /// Is this result trustworthy enough to act on?
    pub fn is_actionable(&self) -> bool {
        self.soundness != SoundnessLevel::Uncertain && self.confidence >= 0.5
    }

    /// Return the epistemic "humility score" — higher means more uncertainty.
    /// Range: 0.0 (fully confident) to 1.0 (maximally uncertain).
    pub fn humility(&self) -> f64 {
        1.0 - self.confidence
    }
}

impl Default for EpistemicMathResult {
    fn default() -> Self {
        Self {
            value: 0.0,
            error_bound: f64::INFINITY,
            confidence: 0.0,
            method_agreement: 0.0,
            caveats: vec!["No computation performed".to_string()],
            soundness: SoundnessLevel::Uncertain,
        }
    }
}

// ─── Epistemic Gating ─────────────────────────────────────────────────────────

/// Gate that applies epistemic constraints to math service outputs.
///
/// Tracks cumulative statistics to detect systematic issues like
/// growing uncertainty or declining verification rates.
#[derive(Debug, Clone)]
pub struct EpistemicGate {
    /// Running count of verified results.
    verified_count: usize,
    /// Running count of probable results.
    probable_count: usize,
    /// Running count of uncertain results.
    uncertain_count: usize,
    /// Running sum of confidence values (for average).
    confidence_sum: f64,
    /// Total results processed.
    total_count: usize,
}

impl EpistemicGate {
    /// Create a new epistemic gate.
    pub fn new() -> Self {
        Self {
            verified_count: 0,
            probable_count: 0,
            uncertain_count: 0,
            confidence_sum: 0.0,
            total_count: 0,
        }
    }

    /// Record an epistemic result and update running statistics.
    pub fn record(&mut self, result: &EpistemicMathResult) {
        match result.soundness {
            SoundnessLevel::Verified => self.verified_count += 1,
            SoundnessLevel::Probable => self.probable_count += 1,
            SoundnessLevel::Uncertain => self.uncertain_count += 1,
        }
        self.confidence_sum += result.confidence;
        self.total_count += 1;
    }

    /// Average confidence across all recorded results (0.0 if none).
    pub fn average_confidence(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.confidence_sum / self.total_count as f64
    }

    /// Fraction of results that are uncertain (0.0-1.0).
    pub fn uncertainty_rate(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.uncertain_count as f64 / self.total_count as f64
    }

    /// Fraction of results that are verified (0.0-1.0).
    pub fn verification_rate(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.verified_count as f64 / self.total_count as f64
    }

    /// Number of uncertain results recorded.
    pub fn uncertain_count(&self) -> usize {
        self.uncertain_count
    }

    /// Total results processed.
    pub fn total_count(&self) -> usize {
        self.total_count
    }
}

impl Default for EpistemicGate {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_verified_level() {
        let result = EpistemicMathResult::verified(2.0, 1e-12);
        assert_eq!(result.soundness, SoundnessLevel::Verified);
        assert!(result.confidence >= 0.90);
        assert!(result.is_actionable());
        assert!(result.caveats.is_empty());
    }

    #[test]
    fn test_epistemic_probable_level() {
        let result = EpistemicMathResult::probable(3.14159, 1e-8);
        assert_eq!(result.soundness, SoundnessLevel::Probable);
        assert!(result.confidence >= 0.5);
        assert!(result.is_actionable());
        // Should have a caveat about not being verified
        assert!(
            result
                .caveats
                .iter()
                .any(|c| c.contains("not independently verified"))
        );
    }

    #[test]
    fn test_epistemic_uncertain_level() {
        let result = EpistemicMathResult::uncertain(1.0, 0.5);
        assert_eq!(result.soundness, SoundnessLevel::Uncertain);
        assert!(!result.is_actionable());
        // Should have caveats about error and no verification
        assert!(!result.caveats.is_empty());
    }

    #[test]
    fn test_error_bound_propagation() {
        // Small error → Probable
        let small_err = EpistemicMathResult::from_computation(42.0, 1e-8, false, 2, 1, 0.8);
        assert!(small_err.error_bound < 1e-6);
        assert!(small_err.soundness == SoundnessLevel::Probable);

        // Large error → Uncertain
        let large_err = EpistemicMathResult::from_computation(42.0, 1.0, false, 2, 1, 0.8);
        assert_eq!(large_err.soundness, SoundnessLevel::Uncertain);
        assert!(
            large_err
                .caveats
                .iter()
                .any(|c| c.contains("Numerical error"))
        );
    }

    #[test]
    fn test_caveat_generation() {
        // NaN result should generate NaN caveat
        let nan_result = EpistemicMathResult::from_computation(f64::NAN, 0.0, false, 1, 0, 0.1);
        assert!(nan_result.caveats.iter().any(|c| c.contains("NaN")));

        // Infinite result
        let inf_result =
            EpistemicMathResult::from_computation(f64::INFINITY, 0.0, false, 1, 0, 0.1);
        assert!(inf_result.caveats.iter().any(|c| c.contains("infinite")));

        // Low method agreement
        let low_agree = EpistemicMathResult::from_computation(1.0, 1e-8, false, 4, 1, 0.7);
        assert!(
            low_agree
                .caveats
                .iter()
                .any(|c| c.contains("Low method agreement"))
        );

        // Single method
        let single = EpistemicMathResult::from_computation(1.0, 1e-8, false, 1, 1, 0.8);
        assert!(single.caveats.iter().any(|c| c.contains("Only one method")));
    }

    #[test]
    fn test_method_agreement_score() {
        // All methods agree
        let all_agree = EpistemicMathResult::from_computation(1.0, 1e-12, true, 3, 3, 0.99);
        assert!((all_agree.method_agreement - 1.0).abs() < 1e-10);

        // Half agree
        let half_agree = EpistemicMathResult::from_computation(1.0, 1e-8, false, 4, 2, 0.8);
        assert!((half_agree.method_agreement - 0.5).abs() < 1e-10);

        // None agree (zero of zero)
        let none = EpistemicMathResult::from_computation(1.0, 1.0, false, 0, 0, 0.1);
        assert!((none.method_agreement - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_soundness_confidence_floor() {
        assert!((SoundnessLevel::Verified.confidence_floor() - 0.95).abs() < 1e-10);
        assert!((SoundnessLevel::Probable.confidence_floor() - 0.70).abs() < 1e-10);
        assert!((SoundnessLevel::Uncertain.confidence_floor() - 0.30).abs() < 1e-10);
    }

    #[test]
    fn test_humility_score() {
        let verified = EpistemicMathResult::verified(1.0, 1e-15);
        assert!(verified.humility() < 0.1); // Very confident → low humility

        let uncertain = EpistemicMathResult::uncertain(1.0, 1.0);
        assert!(uncertain.humility() > 0.5); // Low confidence → high humility
    }

    #[test]
    fn test_default_epistemic_result() {
        let result = EpistemicMathResult::default();
        assert_eq!(result.soundness, SoundnessLevel::Uncertain);
        assert_eq!(result.confidence, 0.0);
        assert!(!result.is_actionable());
    }

    #[test]
    fn test_epistemic_gate_tracking() {
        let mut gate = EpistemicGate::new();

        gate.record(&EpistemicMathResult::verified(1.0, 1e-15));
        gate.record(&EpistemicMathResult::probable(2.0, 1e-8));
        gate.record(&EpistemicMathResult::uncertain(3.0, 1.0));

        assert_eq!(gate.total_count(), 3);
        assert_eq!(gate.uncertain_count(), 1);
        assert!((gate.uncertainty_rate() - 1.0 / 3.0).abs() < 1e-10);
        assert!((gate.verification_rate() - 1.0 / 3.0).abs() < 1e-10);
        assert!(gate.average_confidence() > 0.0);
    }

    #[test]
    fn test_epistemic_gate_empty() {
        let gate = EpistemicGate::new();
        assert_eq!(gate.total_count(), 0);
        assert_eq!(gate.average_confidence(), 0.0);
        assert_eq!(gate.uncertainty_rate(), 0.0);
    }
}
