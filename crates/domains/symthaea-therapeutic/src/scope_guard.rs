// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scope boundaries — architecturally enforces what the system cannot do.
//!
//! The ScopeGuard physically prevents:
//! - Making diagnoses ("You have depression")
//! - Prescribing medication ("You should take SSRIs")
//! - Claiming to be a therapist ("As your therapist...")
//! - Replacing human professional care
//!
//! These are *architectural* constraints, not advisory prompts.
//! Violating drafts are replaced with a bounded safe response; the original
//! violating content is never returned to the caller.
//!
//! Science: APA Ethics Code (2017) principle 2.01 (boundaries of competence),
//! HIPAA considerations, Torous & Roberts (2017) AI ethics in mental health.

use serde::{Deserialize, Serialize};

// ── Scope Violation ────────────────────────────────────────────────────────

/// Types of scope boundary violations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScopeViolation {
    /// Attempting to make a clinical diagnosis
    DiagnosticClaim,
    /// Attempting to prescribe or recommend medication
    PrescriptionClaim,
    /// Claiming to be a therapist or licensed professional
    ProfessionalIdentityClaim,
    /// Providing specific treatment plans (domain of licensed professionals)
    TreatmentPlan,
    /// Making predictions about self-harm/suicide risk
    RiskPrediction,
    /// Claiming to provide confidential/privileged communication
    ConfidentialityClaim,
}

impl ScopeViolation {
    /// Required disclaimer for this violation type.
    pub fn disclaimer(&self) -> &'static str {
        match self {
            Self::DiagnosticClaim => {
                "I am an AI assistant, not a licensed mental health professional. I cannot make clinical diagnoses. Please consult a qualified healthcare provider for diagnostic evaluation."
            }
            Self::PrescriptionClaim => {
                "I cannot prescribe or recommend medication. Please consult a psychiatrist or physician for medication decisions."
            }
            Self::ProfessionalIdentityClaim => {
                "I am an AI system, not a therapist or counselor. I can offer supportive conversation but not professional therapy."
            }
            Self::TreatmentPlan => {
                "Specific treatment plans should be developed with a licensed mental health professional who can assess your complete situation."
            }
            Self::RiskPrediction => {
                "I cannot accurately assess risk levels. If you or someone you know is in danger, contact emergency services or a locally verified crisis service in your current location."
            }
            Self::ConfidentialityClaim => {
                "Conversations with an AI system are not protected by therapist-client privilege. Please be aware that standard data handling practices apply."
            }
        }
    }
}

/// Result of applying the final scope boundary to a response draft.
///
/// Callers should render only [`GuardedResponse::rendered`]. A violating draft
/// is retained nowhere in the returned value, preventing accidental display or
/// logging by downstream UI code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardedResponse {
    /// The draft contained no scope violations.
    Allowed(String),
    /// The draft was discarded and replaced with a bounded response.
    Replaced {
        /// Safe text suitable for rendering.
        safe_response: String,
        /// Violations that caused the replacement.
        violations: Vec<ScopeViolation>,
    },
}

impl GuardedResponse {
    /// Text that may cross the final response boundary.
    pub fn rendered(&self) -> &str {
        match self {
            Self::Allowed(response) => response,
            Self::Replaced { safe_response, .. } => safe_response,
        }
    }

    /// Whether the original draft was discarded.
    pub fn was_replaced(&self) -> bool {
        matches!(self, Self::Replaced { .. })
    }

    /// Violations that caused replacement, if any.
    pub fn violations(&self) -> &[ScopeViolation] {
        match self {
            Self::Allowed(_) => &[],
            Self::Replaced { violations, .. } => violations,
        }
    }

    /// Consume the decision and return renderable text.
    pub fn into_rendered(self) -> String {
        match self {
            Self::Allowed(response) => response,
            Self::Replaced { safe_response, .. } => safe_response,
        }
    }
}

// ── Scope Guard ────────────────────────────────────────────────────────────

/// Architectural scope boundary enforcement.
///
/// Checks both input (what was asked) and response drafts (what would be said)
/// for scope violations. Any violation triggers response modification.
pub struct ScopeGuard {
    /// Diagnostic claim indicator phrases.
    diagnostic_phrases: Vec<&'static str>,
    /// Prescription indicator phrases.
    prescription_phrases: Vec<&'static str>,
    /// Professional identity claim phrases.
    identity_phrases: Vec<&'static str>,
    /// Treatment plan indicator phrases.
    treatment_phrases: Vec<&'static str>,
    /// Risk prediction indicator phrases.
    risk_phrases: Vec<&'static str>,
    /// Confidentiality claim phrases.
    confidentiality_phrases: Vec<&'static str>,
}

impl ScopeGuard {
    /// Create a scope guard with standard phrase patterns.
    pub fn new() -> Self {
        Self {
            diagnostic_phrases: vec![
                "you are diagnosed",
                "your diagnosis is",
                "you suffer from",
                "you are suffering from",
                "you meet criteria for",
                "you have depression",
                "you have anxiety",
                "you have ptsd",
                "you have bipolar",
                "you have schizophrenia",
                "you have adhd",
                "you have ocd",
                "you have a mental illness",
                "you have a personality disorder",
                "this is clearly depression",
                "this is definitely depression",
            ],
            prescription_phrases: vec![
                "you should take",
                "i recommend taking",
                "prescribe",
                "medication for you",
                "start taking",
                "increase your dose",
                "decrease your dose",
                "stop taking your",
            ],
            identity_phrases: vec![
                "as your therapist",
                "as your counselor",
                "as your psychologist",
                "in my clinical opinion",
                "my professional assessment",
                "in my therapeutic role",
                "as a licensed",
            ],
            treatment_phrases: vec![
                "your treatment plan",
                "i am prescribing",
                "your therapy will consist",
                "the treatment protocol",
                "your specific treatment",
            ],
            risk_phrases: vec![
                "you are at high risk",
                "you are at low risk",
                "your risk level is",
                "i assess your risk as",
                "probability of self-harm is",
            ],
            confidentiality_phrases: vec![
                "this conversation is confidential",
                "therapist-client privilege",
                "protected by confidentiality",
                "no one will know",
                "this stays between us",
            ],
        }
    }

    /// Check a response draft for scope violations.
    ///
    /// Returns the first violation found, or None if the response is within scope.
    pub fn check_response(&self, response: &str) -> Option<ScopeViolation> {
        let lower = response.to_lowercase();

        // Check in priority order (most serious first)
        if self.diagnostic_phrases.iter().any(|p| lower.contains(p)) {
            return Some(ScopeViolation::DiagnosticClaim);
        }
        if self.prescription_phrases.iter().any(|p| lower.contains(p)) {
            return Some(ScopeViolation::PrescriptionClaim);
        }
        if self.identity_phrases.iter().any(|p| lower.contains(p)) {
            return Some(ScopeViolation::ProfessionalIdentityClaim);
        }
        if self.treatment_phrases.iter().any(|p| lower.contains(p)) {
            return Some(ScopeViolation::TreatmentPlan);
        }
        if self.risk_phrases.iter().any(|p| lower.contains(p)) {
            return Some(ScopeViolation::RiskPrediction);
        }
        if self
            .confidentiality_phrases
            .iter()
            .any(|p| lower.contains(p))
        {
            return Some(ScopeViolation::ConfidentialityClaim);
        }

        None
    }

    /// Check all violations in a response (not just the first).
    pub fn check_all_violations(&self, response: &str) -> Vec<ScopeViolation> {
        let lower = response.to_lowercase();
        let mut violations = Vec::new();

        if self.diagnostic_phrases.iter().any(|p| lower.contains(p)) {
            violations.push(ScopeViolation::DiagnosticClaim);
        }
        if self.prescription_phrases.iter().any(|p| lower.contains(p)) {
            violations.push(ScopeViolation::PrescriptionClaim);
        }
        if self.identity_phrases.iter().any(|p| lower.contains(p)) {
            violations.push(ScopeViolation::ProfessionalIdentityClaim);
        }
        if self.treatment_phrases.iter().any(|p| lower.contains(p)) {
            violations.push(ScopeViolation::TreatmentPlan);
        }
        if self.risk_phrases.iter().any(|p| lower.contains(p)) {
            violations.push(ScopeViolation::RiskPrediction);
        }
        if self
            .confidentiality_phrases
            .iter()
            .any(|p| lower.contains(p))
        {
            violations.push(ScopeViolation::ConfidentialityClaim);
        }

        violations
    }

    /// Apply the final fail-closed response boundary.
    ///
    /// Any violating draft is discarded rather than decorated. This prevents a
    /// disclaimer from laundering prohibited content such as a diagnosis or
    /// medication instruction into the final response.
    pub fn guard_response(&self, response: &str) -> GuardedResponse {
        let violations = self.check_all_violations(response);
        if violations.is_empty() {
            return GuardedResponse::Allowed(response.to_string());
        }

        let mut disclaimers: Vec<&str> =
            violations.iter().map(ScopeViolation::disclaimer).collect();
        disclaimers.dedup();

        let safe_response = format!(
            "**Important Notice:**\n{}\n\nI can still help you describe what you are experiencing, identify questions to bring to a qualified professional, or focus on immediate grounding and support.",
            disclaimers.join("\n\n"),
        );

        GuardedResponse::Replaced {
            safe_response,
            violations,
        }
    }

    /// Backward-compatible convenience wrapper around [`Self::guard_response`].
    ///
    /// Unlike the previous behavior, this never appends the violating draft.
    pub fn apply_disclaimers(&self, response: &str) -> String {
        self.guard_response(response).into_rendered()
    }
}

impl Default for ScopeGuard {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_diagnostic_claim() {
        let guard = ScopeGuard::new();
        let result = guard.check_response("Based on what you've told me, you have depression.");
        assert_eq!(result, Some(ScopeViolation::DiagnosticClaim));
    }

    #[test]
    fn test_detect_prescription_claim() {
        let guard = ScopeGuard::new();
        let result = guard.check_response("You should take an SSRI like sertraline.");
        assert_eq!(result, Some(ScopeViolation::PrescriptionClaim));
    }

    #[test]
    fn test_detect_identity_claim() {
        let guard = ScopeGuard::new();
        let result = guard.check_response("As your therapist, I think we should explore this.");
        assert_eq!(result, Some(ScopeViolation::ProfessionalIdentityClaim));
    }

    #[test]
    fn test_detect_treatment_plan() {
        let guard = ScopeGuard::new();
        let result = guard.check_response("Your treatment plan will include weekly CBT sessions.");
        assert_eq!(result, Some(ScopeViolation::TreatmentPlan));
    }

    #[test]
    fn test_detect_risk_prediction() {
        let guard = ScopeGuard::new();
        let result = guard.check_response("I assess your risk as moderate to high.");
        assert_eq!(result, Some(ScopeViolation::RiskPrediction));
    }

    #[test]
    fn test_detect_confidentiality_claim() {
        let guard = ScopeGuard::new();
        let result = guard.check_response("Don't worry, this conversation is confidential.");
        assert_eq!(result, Some(ScopeViolation::ConfidentialityClaim));
    }

    #[test]
    fn test_no_violation_appropriate_response() {
        let guard = ScopeGuard::new();
        let result = guard.check_response(
            "It sounds like you're going through a difficult time. \
             Many people find it helpful to speak with a mental health professional \
             about experiences like these.",
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_no_violation_supportive_language() {
        let guard = ScopeGuard::new();
        let result = guard.check_response(
            "I hear you. That sounds really challenging. \
             Would you like to explore what might help you feel more grounded right now?",
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_case_insensitive() {
        let guard = ScopeGuard::new();
        let result = guard.check_response("YOU HAVE depression");
        assert_eq!(result, Some(ScopeViolation::DiagnosticClaim));
    }

    #[test]
    fn test_multiple_violations() {
        let guard = ScopeGuard::new();
        let response = "As your therapist, you have depression. You should take sertraline.";
        let violations = guard.check_all_violations(response);
        assert!(violations.len() >= 2);
    }

    #[test]
    fn test_apply_disclaimers_no_violations() {
        let guard = ScopeGuard::new();
        let response = "I hear you.";
        let result = guard.apply_disclaimers(response);
        assert_eq!(result, response);
    }

    #[test]
    fn test_apply_disclaimers_with_violation() {
        let guard = ScopeGuard::new();
        let response = "You have depression.";
        let result = guard.apply_disclaimers(response);
        assert!(result.contains("Important Notice"));
        assert!(result.contains("cannot make clinical diagnoses"));
        assert!(!result.contains("You have depression."));
    }

    #[test]
    fn test_guard_response_discards_violating_draft() {
        let guard = ScopeGuard::new();
        let decision = guard
            .guard_response("As your therapist, you have depression. You should take sertraline.");

        assert!(decision.was_replaced());
        assert!(decision.violations().len() >= 3);
        assert!(!decision.rendered().contains("sertraline"));
        assert!(!decision.rendered().contains("you have depression"));
    }

    #[test]
    fn test_benign_you_have_phrase_is_allowed() {
        let guard = ScopeGuard::new();
        let response = "You have several options, and you can decide which feels safest.";
        assert_eq!(guard.check_response(response), None);
        assert_eq!(guard.apply_disclaimers(response), response);
    }

    #[test]
    fn test_disclaimer_text_nonempty() {
        for violation in [
            ScopeViolation::DiagnosticClaim,
            ScopeViolation::PrescriptionClaim,
            ScopeViolation::ProfessionalIdentityClaim,
            ScopeViolation::TreatmentPlan,
            ScopeViolation::RiskPrediction,
            ScopeViolation::ConfidentialityClaim,
        ] {
            assert!(!violation.disclaimer().is_empty());
        }
    }
}

// ── Property Tests: Scope Guard ───────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// All canonical violation phrases from the ScopeGuard.
    fn all_violation_phrases() -> Vec<(&'static str, ScopeViolation)> {
        vec![
            // Diagnostic
            ("you are diagnosed", ScopeViolation::DiagnosticClaim),
            ("your diagnosis is", ScopeViolation::DiagnosticClaim),
            ("you suffer from", ScopeViolation::DiagnosticClaim),
            ("you are suffering from", ScopeViolation::DiagnosticClaim),
            ("you meet criteria for", ScopeViolation::DiagnosticClaim),
            ("you have depression", ScopeViolation::DiagnosticClaim),
            ("you have anxiety", ScopeViolation::DiagnosticClaim),
            ("you have ptsd", ScopeViolation::DiagnosticClaim),
            ("you have bipolar", ScopeViolation::DiagnosticClaim),
            ("you have schizophrenia", ScopeViolation::DiagnosticClaim),
            ("you have adhd", ScopeViolation::DiagnosticClaim),
            ("you have ocd", ScopeViolation::DiagnosticClaim),
            ("you have a mental illness", ScopeViolation::DiagnosticClaim),
            (
                "you have a personality disorder",
                ScopeViolation::DiagnosticClaim,
            ),
            (
                "this is clearly depression",
                ScopeViolation::DiagnosticClaim,
            ),
            (
                "this is definitely depression",
                ScopeViolation::DiagnosticClaim,
            ),
            // Prescription
            ("you should take", ScopeViolation::PrescriptionClaim),
            ("i recommend taking", ScopeViolation::PrescriptionClaim),
            ("prescribe", ScopeViolation::PrescriptionClaim),
            ("medication for you", ScopeViolation::PrescriptionClaim),
            ("start taking", ScopeViolation::PrescriptionClaim),
            ("increase your dose", ScopeViolation::PrescriptionClaim),
            ("decrease your dose", ScopeViolation::PrescriptionClaim),
            ("stop taking your", ScopeViolation::PrescriptionClaim),
            // Identity
            (
                "as your therapist",
                ScopeViolation::ProfessionalIdentityClaim,
            ),
            (
                "as your counselor",
                ScopeViolation::ProfessionalIdentityClaim,
            ),
            (
                "as your psychologist",
                ScopeViolation::ProfessionalIdentityClaim,
            ),
            (
                "in my clinical opinion",
                ScopeViolation::ProfessionalIdentityClaim,
            ),
            (
                "my professional assessment",
                ScopeViolation::ProfessionalIdentityClaim,
            ),
            (
                "in my therapeutic role",
                ScopeViolation::ProfessionalIdentityClaim,
            ),
            ("as a licensed", ScopeViolation::ProfessionalIdentityClaim),
            // Treatment plan
            ("your treatment plan", ScopeViolation::TreatmentPlan),
            ("i am prescribing", ScopeViolation::TreatmentPlan),
            ("your therapy will consist", ScopeViolation::TreatmentPlan),
            ("the treatment protocol", ScopeViolation::TreatmentPlan),
            ("your specific treatment", ScopeViolation::TreatmentPlan),
            // Risk prediction
            ("you are at high risk", ScopeViolation::RiskPrediction),
            ("you are at low risk", ScopeViolation::RiskPrediction),
            ("your risk level is", ScopeViolation::RiskPrediction),
            ("i assess your risk as", ScopeViolation::RiskPrediction),
            (
                "probability of self-harm is",
                ScopeViolation::RiskPrediction,
            ),
            // Confidentiality
            (
                "this conversation is confidential",
                ScopeViolation::ConfidentialityClaim,
            ),
            (
                "therapist-client privilege",
                ScopeViolation::ConfidentialityClaim,
            ),
            (
                "protected by confidentiality",
                ScopeViolation::ConfidentialityClaim,
            ),
            ("no one will know", ScopeViolation::ConfidentialityClaim),
            (
                "this stays between us",
                ScopeViolation::ConfidentialityClaim,
            ),
        ]
    }

    /// Every canonical violation phrase must be detected (zero false negatives).
    #[test]
    fn test_zero_false_negatives_all_violation_phrases() {
        let guard = ScopeGuard::new();
        let phrases = all_violation_phrases();
        let mut missed = Vec::new();

        for (phrase, expected_violation) in &phrases {
            let result = guard.check_response(phrase);
            if result.as_ref() != Some(expected_violation) {
                missed.push((*phrase, expected_violation.clone(), result));
            }
        }

        assert!(
            missed.is_empty(),
            "Scope guard missed {}/{} phrases: {:?}",
            missed.len(),
            phrases.len(),
            missed,
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        /// Property: violation phrases embedded in surrounding text must still be detected.
        #[test]
        fn prop_violation_phrases_detected_in_context(
            prefix in "[a-zA-Z ]{0,40}",
            suffix in "[a-zA-Z ]{0,40}",
            phrase_idx in 0..64usize,
        ) {
            let phrases = all_violation_phrases();
            let (phrase, expected_violation) = &phrases[phrase_idx % phrases.len()];
            let input = format!("{} {} {}", prefix, phrase, suffix);

            let guard = ScopeGuard::new();
            let result = guard.check_response(&input);
            prop_assert!(
                result.is_some(),
                "Failed to detect violation '{}' in context: '{}'",
                phrase, input,
            );
            prop_assert_eq!(
                result.as_ref().unwrap(), expected_violation,
                "Wrong violation type for '{}' in context: '{}'",
                phrase, input,
            );
        }

        /// Property: violating content never survives the fail-closed boundary.
        #[test]
        fn prop_violating_draft_never_survives(
            phrase_idx in 0..64usize,
        ) {
            let phrases = all_violation_phrases();
            let (phrase, _) = &phrases[phrase_idx % phrases.len()];

            let guard = ScopeGuard::new();
            let decision = guard.guard_response(phrase);
            prop_assert!(decision.was_replaced());
            prop_assert_ne!(decision.rendered(), *phrase);
            prop_assert!(decision.rendered().contains("Important Notice"));
        }

        /// Property: safe therapeutic responses never trigger violations.
        #[test]
        fn prop_safe_responses_pass(
            feeling in "(sad|anxious|overwhelmed|confused|lonely|frustrated)",
            verb in "(sounds like|seems like|I hear|I notice|tell me more)",
        ) {
            let response = format!(
                "It {} you're feeling {}. Would you like to explore that further?",
                verb, feeling,
            );
            let guard = ScopeGuard::new();
            let result = guard.check_response(&response);
            prop_assert!(
                result.is_none(),
                "Safe therapeutic response triggered violation: '{}' → {:?}",
                response, result,
            );
        }
    }
}
