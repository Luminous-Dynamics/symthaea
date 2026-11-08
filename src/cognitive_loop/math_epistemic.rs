#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Numerical Epistemic Gating for Mathematical Outputs (Phase 7b)
//!
//! Applies epistemic constraints to numerical math outputs so that Symthaea
//! knows what its computations do and do not establish.
//!
//! Every result carries an [`EpistemicMathResult`] with:
//! - numerical error bounds;
//! - method-count and cross-method agreement evidence;
//! - a numerical evidence classification;
//! - explicit caveats ("what I don't know").
//!
//! ## Authority boundary
//!
//! Numerical agreement is evidence about a computation, not theorem authority.
//!
//! ```text
//! multiple numerical methods agree
//!     != formal proof
//!     != Lean/kernel acceptance
//!     != specification correctness
//!     != mathematical novelty
//! ```
//!
//! Formal authority belongs to the dedicated proof/specification evidence stack.
//! This module deliberately cannot mint it.
//!
//! Science: Lakatos (1976) — mathematical knowledge is fallible and progresses
//! through proofs and refutations. Polya (1954) — plausible mathematical
//! reasoning requires tracking uncertainty without confusing plausibility with
//! proof.

use serde::{Deserialize, Serialize};

// ─── Numerical Evidence Level ─────────────────────────────────────────────────

/// Classification of evidence supporting a numerical mathematical output.
///
/// This is intentionally **not** a theorem-soundness or proof-verification
/// classification. The strongest state means that multiple reported numerical
/// methods agree within a tight declared error bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericalEvidenceLevel {
    /// Multiple methods corroborate a finite result with a tight finite error
    /// bound. This remains numerical evidence only.
    #[serde(alias = "Verified")]
    Corroborated,
    /// The result has some usable numerical support but lacks qualified
    /// multi-method corroboration.
    #[serde(alias = "Probable")]
    Plausible,
    /// Evidence is missing, internally inconsistent, non-finite, or too weak.
    Uncertain,
}

impl NumericalEvidenceLevel {
    /// Human-readable evidence label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Corroborated => "NumericallyCorroborated",
            Self::Plausible => "Plausible",
            Self::Uncertain => "Uncertain",
        }
    }

    /// Maximum confidence compatible with this numerical evidence tier.
    ///
    /// This is a ceiling, never a floor: evidence classification may reduce a
    /// solver's declared confidence but may not manufacture confidence that the
    /// solver itself did not have.
    pub fn confidence_ceiling(&self) -> f64 {
        match self {
            Self::Corroborated => 0.99,
            Self::Plausible => 0.85,
            Self::Uncertain => 0.49,
        }
    }
}

impl Default for NumericalEvidenceLevel {
    fn default() -> Self {
        Self::Uncertain
    }
}

/// Compatibility name for downstream callers during the terminology migration.
///
/// New code should use [`NumericalEvidenceLevel`]. The historical name could be
/// misread as formal mathematical soundness.
#[deprecated(
    note = "use NumericalEvidenceLevel; this layer classifies numerical evidence, not formal soundness"
)]
pub type SoundnessLevel = NumericalEvidenceLevel;

// ─── Epistemic Math Result ────────────────────────────────────────────────────

/// A numerical math result with explicit epistemic metadata.
///
/// The result describes numerical evidence only. It cannot carry or imply a
/// formal proof receipt. Public/Serde fields are treated as evidence to
/// re-validate, never as self-authorizing classification state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicMathResult {
    /// The computed value.
    pub value: f64,
    /// Estimated numerical error bound (e.g. floating-point/truncation error).
    pub error_bound: f64,
    /// Solver confidence after fail-closed numerical-evidence capping (0.0-1.0).
    pub confidence: f64,
    /// Fraction of distinct tried methods that agree when at least two methods
    /// were tried. A single method has agreement 0.0 because it provides no
    /// cross-method corroboration.
    pub method_agreement: f64,
    /// Total numerical methods reported for this result.
    #[serde(default)]
    pub method_count: usize,
    /// Methods reported to agree within the caller's declared tolerance.
    #[serde(default)]
    pub agreeing_methods: usize,
    /// Whether the caller reports that a genuine multipath comparison occurred.
    /// This flag is checked against the method counts and cannot establish formal
    /// verification authority.
    #[serde(default, alias = "multipath_verified")]
    pub multipath_reported: bool,
    /// Explicit caveats: things the numerical computation does not establish.
    pub caveats: Vec<String>,
    /// Stored numerical evidence classification. Promotion consumers re-derive
    /// this from the raw evidence fields before trusting it.
    #[serde(alias = "soundness")]
    pub numerical_evidence: NumericalEvidenceLevel,
}

impl EpistemicMathResult {
    /// Classify numerical evidence from the complete reported computation state.
    ///
    /// Invalid or contradictory metadata always fails closed to `Uncertain`.
    pub fn classify_numerical_evidence(
        value: f64,
        error_bound: f64,
        multipath_reported: bool,
        method_count: usize,
        agreeing_methods: usize,
    ) -> NumericalEvidenceLevel {
        if !Self::evidence_inputs_valid(
            value,
            error_bound,
            multipath_reported,
            method_count,
            agreeing_methods,
        ) {
            return NumericalEvidenceLevel::Uncertain;
        }

        let method_agreement = Self::cross_method_agreement(method_count, agreeing_methods);
        let multipath_supported =
            multipath_reported && method_count >= 2 && agreeing_methods >= 2;

        if multipath_supported && error_bound < 1e-10 && method_agreement >= 0.9 {
            NumericalEvidenceLevel::Corroborated
        } else if error_bound < 1e-6 && agreeing_methods >= 1 {
            NumericalEvidenceLevel::Plausible
        } else {
            NumericalEvidenceLevel::Uncertain
        }
    }

    /// Historical classification entry point retained during migration.
    ///
    /// Because the old signature did not contain method counts or the numerical
    /// value, it cannot establish the strongest tier anymore. Callers that need
    /// full classification should use [`classify_numerical_evidence`](Self::classify_numerical_evidence).
    #[deprecated(
        note = "use classify_numerical_evidence with value and method counts; old inputs cannot establish corroboration"
    )]
    pub fn classify_soundness(
        error_bound: f64,
        method_agreement: f64,
        _multipath_reported: bool,
    ) -> NumericalEvidenceLevel {
        if !error_bound.is_finite()
            || error_bound < 0.0
            || !method_agreement.is_finite()
            || !(0.0..=1.0).contains(&method_agreement)
        {
            return NumericalEvidenceLevel::Uncertain;
        }

        // The historical interface has no method-count or value evidence, so it
        // is intentionally incapable of establishing the strongest tier.
        if error_bound < 1e-6 && method_agreement > 0.0 {
            NumericalEvidenceLevel::Plausible
        } else {
            NumericalEvidenceLevel::Uncertain
        }
    }

    /// Create an epistemic result from a raw numerical computation.
    ///
    /// `multipath_reported` is treated as a declaration to validate, not as an
    /// authority token. At least two methods and two agreeing methods are
    /// required before it can contribute to the strongest numerical tier.
    pub fn from_computation(
        value: f64,
        error_bound: f64,
        multipath_reported: bool,
        method_count: usize,
        agreeing_methods: usize,
        base_confidence: f64,
    ) -> Self {
        let evidence_inputs_valid = Self::evidence_inputs_valid(
            value,
            error_bound,
            multipath_reported,
            method_count,
            agreeing_methods,
        );

        let method_agreement = Self::cross_method_agreement(method_count, agreeing_methods);
        let numerical_evidence = Self::classify_numerical_evidence(
            value,
            error_bound,
            multipath_reported,
            method_count,
            agreeing_methods,
        );

        let base_confidence_valid =
            base_confidence.is_finite() && (0.0..=1.0).contains(&base_confidence);
        let confidence = if evidence_inputs_valid && base_confidence_valid {
            base_confidence.min(numerical_evidence.confidence_ceiling())
        } else {
            0.0
        };

        let mut caveats = Vec::new();

        if !value.is_finite() {
            caveats.push("Result is non-finite; numerical evidence fails closed".to_string());
        }
        if !error_bound.is_finite() || error_bound < 0.0 {
            caveats.push("Error bound must be finite and non-negative".to_string());
        } else if error_bound > 1e-6 {
            caveats.push(format!(
                "Numerical error may be significant (bound: {:.2e})",
                error_bound
            ));
        }
        if method_count == 0 {
            caveats.push("No numerical method was recorded".to_string());
        }
        if agreeing_methods > method_count {
            caveats.push(format!(
                "Invalid method counts: {} agreeing methods exceeds {} tried methods",
                agreeing_methods, method_count
            ));
        }
        if method_count == 1 {
            caveats.push("Only one method available; no cross-method corroboration possible".to_string());
        }
        if multipath_reported && (method_count < 2 || agreeing_methods < 2) {
            caveats.push(
                "Multipath corroboration was reported without at least two agreeing methods"
                    .to_string(),
            );
        } else if !multipath_reported {
            caveats.push("No qualified multi-method corroboration reported".to_string());
        }
        if method_count > 1 && method_agreement < 0.5 {
            caveats.push(format!(
                "Low cross-method agreement ({}/{} methods agree)",
                agreeing_methods, method_count
            ));
        }
        if !base_confidence_valid {
            caveats.push("Base confidence must be finite and within [0, 1]".to_string());
        }

        Self {
            value,
            error_bound,
            confidence,
            method_agreement,
            method_count,
            agreeing_methods,
            multipath_reported,
            caveats,
            numerical_evidence,
        }
    }

    /// Create a strongly numerically corroborated result.
    pub fn corroborated(value: f64, error_bound: f64) -> Self {
        Self::from_computation(value, error_bound, true, 3, 3, 0.99)
    }

    /// Historical constructor retained as a compatibility shim.
    #[deprecated(note = "use corroborated(); numerical corroboration is not formal verification")]
    pub fn verified(value: f64, error_bound: f64) -> Self {
        Self::corroborated(value, error_bound)
    }

    /// Create a plausible single-method numerical result.
    pub fn plausible(value: f64, error_bound: f64) -> Self {
        Self::from_computation(value, error_bound, false, 1, 1, 0.80)
    }

    /// Historical constructor retained as a compatibility shim.
    #[deprecated(note = "use plausible()")]
    pub fn probable(value: f64, error_bound: f64) -> Self {
        Self::plausible(value, error_bound)
    }

    /// Create an uncertain result.
    pub fn uncertain(value: f64, error_bound: f64) -> Self {
        Self::from_computation(value, error_bound, false, 1, 0, 0.30)
    }

    /// Re-derive the evidence level from raw fields and fail closed on any
    /// self-inconsistency in the stored classification/agreement/confidence.
    pub fn effective_numerical_evidence(&self) -> NumericalEvidenceLevel {
        if self.is_self_consistent() {
            self.numerical_evidence
        } else {
            NumericalEvidenceLevel::Uncertain
        }
    }

    /// Whether the public/Serde representation is internally consistent with
    /// the evidence rules implemented by this module.
    pub fn is_self_consistent(&self) -> bool {
        let expected_level = Self::classify_numerical_evidence(
            self.value,
            self.error_bound,
            self.multipath_reported,
            self.method_count,
            self.agreeing_methods,
        );
        let expected_agreement =
            Self::cross_method_agreement(self.method_count, self.agreeing_methods);
        let confidence_valid = self.confidence.is_finite()
            && (0.0..=expected_level.confidence_ceiling()).contains(&self.confidence);

        Self::evidence_inputs_valid(
            self.value,
            self.error_bound,
            self.multipath_reported,
            self.method_count,
            self.agreeing_methods,
        ) && self.numerical_evidence == expected_level
            && (self.method_agreement - expected_agreement).abs() <= 1e-12
            && confidence_valid
    }

    /// Whether the numerical output is usable as a bounded computation result.
    ///
    /// This is intentionally weaker than mathematical/formal authority.
    pub fn is_numerically_usable(&self) -> bool {
        self.is_self_consistent()
            && self.effective_numerical_evidence() != NumericalEvidenceLevel::Uncertain
            && self.confidence >= 0.5
    }

    /// Historical convenience name retained during migration.
    #[deprecated(note = "use is_numerically_usable(); numerical usability is not general action authority")]
    pub fn is_actionable(&self) -> bool {
        self.is_numerically_usable()
    }

    /// Numerical evidence in this module never grants formal theorem authority.
    pub const fn grants_formal_authority(&self) -> bool {
        false
    }

    /// Return the epistemic "humility score" — higher means more uncertainty.
    /// Range: 0.0 (fully confident) to 1.0 (maximally uncertain).
    pub fn humility(&self) -> f64 {
        1.0 - self.confidence
    }

    fn evidence_inputs_valid(
        value: f64,
        error_bound: f64,
        multipath_reported: bool,
        method_count: usize,
        agreeing_methods: usize,
    ) -> bool {
        value.is_finite()
            && error_bound.is_finite()
            && error_bound >= 0.0
            && method_count > 0
            && agreeing_methods <= method_count
            && (!multipath_reported || (method_count >= 2 && agreeing_methods >= 2))
    }

    fn cross_method_agreement(method_count: usize, agreeing_methods: usize) -> f64 {
        if method_count < 2 || agreeing_methods > method_count {
            0.0
        } else {
            agreeing_methods as f64 / method_count as f64
        }
    }
}

impl Default for EpistemicMathResult {
    fn default() -> Self {
        Self {
            value: 0.0,
            error_bound: f64::INFINITY,
            confidence: 0.0,
            method_agreement: 0.0,
            method_count: 0,
            agreeing_methods: 0,
            multipath_reported: false,
            caveats: vec!["No computation performed".to_string()],
            numerical_evidence: NumericalEvidenceLevel::Uncertain,
        }
    }
}

// ─── Epistemic Gating ─────────────────────────────────────────────────────────

/// Tracks numerical evidence quality over math-service outputs.
#[derive(Debug, Clone)]
pub struct EpistemicGate {
    /// Running count of numerically corroborated results.
    corroborated_count: usize,
    /// Running count of plausible results.
    plausible_count: usize,
    /// Running count of uncertain results.
    uncertain_count: usize,
    /// Running count of structurally inconsistent/tampered records.
    invalid_count: usize,
    /// Running sum of confidence values (for average; invalid records add zero).
    confidence_sum: f64,
    /// Total results processed.
    total_count: usize,
}

impl EpistemicGate {
    /// Create a new epistemic gate.
    pub fn new() -> Self {
        Self {
            corroborated_count: 0,
            plausible_count: 0,
            uncertain_count: 0,
            invalid_count: 0,
            confidence_sum: 0.0,
            total_count: 0,
        }
    }

    /// Record a numerical epistemic result and update running statistics.
    ///
    /// Stored classification is never trusted directly: it is re-derived from
    /// the raw evidence fields. Invalid records are counted as uncertain and add
    /// zero confidence.
    pub fn record(&mut self, result: &EpistemicMathResult) {
        let consistent = result.is_self_consistent();
        match result.effective_numerical_evidence() {
            NumericalEvidenceLevel::Corroborated => self.corroborated_count += 1,
            NumericalEvidenceLevel::Plausible => self.plausible_count += 1,
            NumericalEvidenceLevel::Uncertain => self.uncertain_count += 1,
        }
        if consistent {
            self.confidence_sum += result.confidence;
        } else {
            self.invalid_count += 1;
        }
        self.total_count += 1;
    }

    /// Average numerical confidence across all recorded results (0.0 if none).
    /// Invalid records remain in the denominator and contribute zero.
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

    /// Fraction of results with strong multi-method numerical corroboration.
    pub fn corroboration_rate(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.corroborated_count as f64 / self.total_count as f64
    }

    /// Historical metric name retained during migration.
    #[deprecated(note = "use corroboration_rate(); this metric is not formal verification")]
    pub fn verification_rate(&self) -> f64 {
        self.corroboration_rate()
    }

    /// Number of structurally inconsistent/tampered results recorded.
    pub fn invalid_count(&self) -> usize {
        self.invalid_count
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
    fn numerically_corroborated_is_not_formal_authority() {
        let result = EpistemicMathResult::corroborated(2.0, 1e-12);
        assert_eq!(
            result.numerical_evidence,
            NumericalEvidenceLevel::Corroborated
        );
        assert!(result.confidence >= 0.90);
        assert!(result.is_numerically_usable());
        assert!(!result.grants_formal_authority());
        assert!(result.caveats.is_empty());
    }

    #[test]
    fn plausible_single_method_has_no_cross_method_agreement() {
        let result = EpistemicMathResult::plausible(std::f64::consts::PI, 1e-8);
        assert_eq!(result.numerical_evidence, NumericalEvidenceLevel::Plausible);
        assert_eq!(result.method_agreement, 0.0);
        assert!(result.is_numerically_usable());
        assert!(result
            .caveats
            .iter()
            .any(|c| c.contains("Only one method")));
    }

    #[test]
    fn non_finite_values_fail_closed() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result =
                EpistemicMathResult::from_computation(value, 1e-12, true, 3, 3, 0.99);
            assert_eq!(result.numerical_evidence, NumericalEvidenceLevel::Uncertain);
            assert_eq!(result.confidence, 0.0);
            assert!(!result.is_numerically_usable());
        }
    }

    #[test]
    fn invalid_error_bounds_fail_closed() {
        for error_bound in [-1e-12, f64::NAN, f64::INFINITY] {
            let result =
                EpistemicMathResult::from_computation(1.0, error_bound, true, 3, 3, 0.99);
            assert_eq!(result.numerical_evidence, NumericalEvidenceLevel::Uncertain);
            assert_eq!(result.confidence, 0.0);
        }
    }

    #[test]
    fn reported_multipath_requires_multiple_methods() {
        let one_method = EpistemicMathResult::from_computation(1.0, 1e-12, true, 1, 1, 0.99);
        assert_eq!(
            one_method.numerical_evidence,
            NumericalEvidenceLevel::Uncertain
        );
        assert!(one_method
            .caveats
            .iter()
            .any(|c| c.contains("Multipath corroboration was reported")));

        let only_one_agrees =
            EpistemicMathResult::from_computation(1.0, 1e-12, true, 3, 1, 0.99);
        assert_eq!(
            only_one_agrees.numerical_evidence,
            NumericalEvidenceLevel::Uncertain
        );
    }

    #[test]
    fn impossible_method_counts_fail_closed() {
        let result = EpistemicMathResult::from_computation(1.0, 1e-12, false, 2, 3, 0.99);
        assert_eq!(result.numerical_evidence, NumericalEvidenceLevel::Uncertain);
        assert_eq!(result.method_agreement, 0.0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn evidence_tier_never_inflates_solver_confidence() {
        let result = EpistemicMathResult::from_computation(1.0, 1e-12, true, 3, 3, 0.20);
        assert_eq!(
            result.numerical_evidence,
            NumericalEvidenceLevel::Corroborated
        );
        assert!((result.confidence - 0.20).abs() < 1e-12);
        assert!(!result.is_numerically_usable());
    }

    #[test]
    fn uncertain_tier_caps_high_solver_confidence() {
        let result = EpistemicMathResult::from_computation(42.0, 1.0, false, 1, 1, 0.99);
        assert_eq!(result.numerical_evidence, NumericalEvidenceLevel::Uncertain);
        assert!(result.confidence <= NumericalEvidenceLevel::Uncertain.confidence_ceiling());
        assert!(!result.is_numerically_usable());
    }

    #[test]
    fn tampered_stored_classification_fails_closed() {
        let mut result = EpistemicMathResult::plausible(2.0, 1e-8);
        result.numerical_evidence = NumericalEvidenceLevel::Corroborated;

        assert!(!result.is_self_consistent());
        assert_eq!(
            result.effective_numerical_evidence(),
            NumericalEvidenceLevel::Uncertain
        );
        assert!(!result.is_numerically_usable());
    }

    #[test]
    fn tampered_method_agreement_fails_closed() {
        let mut result = EpistemicMathResult::corroborated(2.0, 1e-12);
        result.method_agreement = 0.5;

        assert!(!result.is_self_consistent());
        assert_eq!(
            result.effective_numerical_evidence(),
            NumericalEvidenceLevel::Uncertain
        );
    }

    #[test]
    fn legacy_missing_method_evidence_cannot_retain_corroboration() {
        let mut result = EpistemicMathResult::corroborated(2.0, 1e-12);
        result.method_count = 0;
        result.agreeing_methods = 0;
        result.multipath_reported = false;

        assert!(!result.is_self_consistent());
        assert_eq!(
            result.effective_numerical_evidence(),
            NumericalEvidenceLevel::Uncertain
        );
        assert!(!result.is_numerically_usable());
    }

    #[test]
    fn low_cross_method_agreement_is_explicit() {
        let result = EpistemicMathResult::from_computation(1.0, 1e-8, false, 4, 1, 0.7);
        assert!(result
            .caveats
            .iter()
            .any(|c| c.contains("Low cross-method agreement")));
    }

    #[test]
    fn confidence_ceiling_is_monotone_with_evidence_strength() {
        assert!(
            NumericalEvidenceLevel::Corroborated.confidence_ceiling()
                > NumericalEvidenceLevel::Plausible.confidence_ceiling()
        );
        assert!(
            NumericalEvidenceLevel::Plausible.confidence_ceiling()
                > NumericalEvidenceLevel::Uncertain.confidence_ceiling()
        );
    }

    #[test]
    fn default_epistemic_result_is_non_authorizing() {
        let result = EpistemicMathResult::default();
        assert_eq!(result.numerical_evidence, NumericalEvidenceLevel::Uncertain);
        assert_eq!(result.confidence, 0.0);
        assert!(!result.is_numerically_usable());
        assert!(!result.grants_formal_authority());
    }

    #[test]
    fn epistemic_gate_rederives_evidence_and_tracks_invalid_records() {
        let mut gate = EpistemicGate::new();

        gate.record(&EpistemicMathResult::corroborated(1.0, 1e-15));
        gate.record(&EpistemicMathResult::plausible(2.0, 1e-8));
        gate.record(&EpistemicMathResult::uncertain(3.0, 1.0));

        let mut tampered = EpistemicMathResult::plausible(4.0, 1e-8);
        tampered.numerical_evidence = NumericalEvidenceLevel::Corroborated;
        gate.record(&tampered);

        assert_eq!(gate.total_count(), 4);
        assert_eq!(gate.invalid_count(), 1);
        assert_eq!(gate.uncertain_count(), 2);
        assert!((gate.uncertainty_rate() - 0.5).abs() < 1e-10);
        assert!((gate.corroboration_rate() - 0.25).abs() < 1e-10);
        assert!(gate.average_confidence() > 0.0);
    }

    #[test]
    fn epistemic_gate_empty() {
        let gate = EpistemicGate::new();
        assert_eq!(gate.total_count(), 0);
        assert_eq!(gate.average_confidence(), 0.0);
        assert_eq!(gate.uncertainty_rate(), 0.0);
        assert_eq!(gate.corroboration_rate(), 0.0);
        assert_eq!(gate.invalid_count(), 0);
    }
}
