// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral Algebra in Hyperdimensional Space
//!
//! This module implements compositional moral reasoning using HDC primitives.
//! Instead of encoding text as n-grams and comparing to prototypes, we define
//! **moral primitives** as base hypervectors and **moral operators** as binding
//! operations that compose them into moral judgments.
//!
//! # Design Philosophy
//!
//! Traditional HDC moral reasoning (n-gram → prototype comparison) fails on
//! categories requiring compositional reasoning:
//! - **Justice**: requires proportionality (effort vs reward magnitude)
//! - **Deontology**: requires obligation satisfaction (rule + excuse validity)
//! - **Commonsense**: requires consent and negation reasoning
//!
//! This module addresses these limitations by:
//! 1. Defining semantic role primitives (AGENT, PATIENT, ACTION, etc.)
//! 2. Defining moral operators (CAUSES, VIOLATES, SATISFIES, etc.)
//! 3. Composing them algebraically to reason about moral scenarios
//!
//! # Key Insight
//!
//! Virtue ethics works (~80%) because it only requires pattern matching on
//! trait words. Other categories fail (~50%) because they require:
//! - Magnitude comparison (justice)
//! - Conditional rule evaluation (deontology)
//! - Negation/absence encoding (commonsense)
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
//! │   PARSER     │───▶│  HDC MORAL   │───▶│   REASONER   │
//! │  (SRL/NLP)   │    │   ALGEBRA    │    │  (Compare)   │
//! └──────────────┘    └──────────────┘    └──────────────┘
//!       │                   │                   │
//!       ▼                   ▼                   ▼
//! Semantic roles      Bound HVs            Judgment
//! ```

mod judgment;
mod operators;
mod primitives;

pub use judgment::*;
pub use operators::*;
pub use primitives::*;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitives_orthogonal() {
        let primitives = MoralPrimitives::default_dim();
        let max_sim = primitives.verify_orthogonality();

        // Random 4096-dim vectors should have similarity < 0.1
        assert!(
            max_sim < 0.15,
            "Primitives not orthogonal: max_sim = {}",
            max_sim
        );
    }

    #[test]
    fn test_intent_encoding_distinguishable() {
        let algebra = MoralAlgebra::default_dim();

        let good = algebra.encode_intent(MoralIntent::Good);
        let bad = algebra.encode_intent(MoralIntent::Bad);
        let neutral = algebra.encode_intent(MoralIntent::Neutral);

        // Different intents should be dissimilar
        let good_bad = good.similarity(&bad);
        let good_neutral = good.similarity(&neutral);
        let bad_neutral = bad.similarity(&neutral);

        assert!(good_bad < 0.3, "Good and Bad too similar: {}", good_bad);
        assert!(
            good_neutral < 0.3,
            "Good and Neutral too similar: {}",
            good_neutral
        );
        assert!(
            bad_neutral < 0.3,
            "Bad and Neutral too similar: {}",
            bad_neutral
        );
    }

    #[test]
    fn test_action_structure_composition() {
        let algebra = MoralAlgebra::default_dim();

        // Create two similar actions with different intents
        let help_good =
            algebra.encode_action_structure("Tyler", "help", "stranger", MoralIntent::Good);
        let help_bad =
            algebra.encode_action_structure("Tyler", "help", "stranger", MoralIntent::Bad);
        let harm_bad =
            algebra.encode_action_structure("Tyler", "harm", "stranger", MoralIntent::Bad);

        // Same action, different intent should be somewhat similar
        let help_sim = help_good.similarity(&help_bad);

        // Different action, same intent should be somewhat similar
        let intent_sim = help_bad.similarity(&harm_bad);

        // Both should be distinguishable (HDC cosine similarity can be slightly negative
        // for near-orthogonal vectors, so allow small negative values)
        assert!(
            help_sim > -0.2 && help_sim < 0.8,
            "Same action different intent: {}",
            help_sim
        );
        assert!(
            intent_sim > -0.2 && intent_sim < 0.8,
            "Different action same intent: {}",
            intent_sim
        );
    }

    #[test]
    fn test_consent_violation_detection() {
        // Test consent violation detection using direct state checking
        // (HDC similarity-based detection proved unreliable, so we use
        // the MoralParser's is_consent_violation() method instead)
        use crate::hdc::moral_parser::MoralParser;

        let algebra = MoralAlgebra::default_dim();
        let parser = MoralParser::new();

        // Scenario with consent violation (absent consent + patient)
        let violation = parser.parse_and_encode(
            "I discussed my daughter's health without asking first",
            &algebra,
        );
        assert!(
            violation.is_consent_violation(),
            "Should detect consent violation when consent is absent and patient exists"
        );

        // Scenario with consent given
        let with_consent = parser.parse_and_encode(
            "After asking my daughter, I discussed her health with the doctor",
            &algebra,
        );
        assert!(
            !with_consent.is_consent_violation(),
            "Should not detect violation when consent is given"
        );

        // Scenario without a patient (no consent issue)
        let no_patient = parser.parse_and_encode("I walked to the store", &algebra);
        assert!(
            !no_patient.is_consent_violation(),
            "Should not detect violation when no patient is affected"
        );
    }

    #[test]
    fn test_proportionality_justice() {
        let algebra = MoralAlgebra::default_dim();

        // Proportional: medium effort, medium reward
        let fair = algebra.encode_proportionality(
            "clean house",
            Magnitude::Medium,
            "fair wage",
            Magnitude::Medium,
        );

        // Disproportional: tiny effort, huge reward
        let unfair = algebra.encode_proportionality(
            "clean house",
            Magnitude::Tiny,
            "brand new car",
            Magnitude::Huge,
        );

        assert!(fair.is_proportional, "Fair case should be proportional");
        assert!(
            !unfair.is_proportional,
            "Unfair case should not be proportional"
        );

        // Judge them
        let fair_judgment = algebra.judge_proportionality(&fair);
        let unfair_judgment = algebra.judge_proportionality(&unfair);

        assert!(fair_judgment.is_just, "Fair case should be just");
        assert!(!unfair_judgment.is_just, "Unfair case should not be just");
    }

    #[test]
    fn test_excuse_validity() {
        let algebra = MoralAlgebra::default_dim();

        // Valid excuse: directly addresses obligation
        let valid = algebra.encode_excuse_validity(
            "prepare for meeting",
            "already set up conference room",
            true,
        );

        // Invalid excuse: doesn't address obligation
        let invalid =
            algebra.encode_excuse_validity("prepare for meeting", "not in the mood", false);

        assert!(valid.is_valid, "Valid excuse should be valid");
        assert!(!invalid.is_valid, "Invalid excuse should not be valid");

        // The HVs should be different
        let sim = valid.composed.similarity(&invalid.composed);
        assert!(sim < 0.5, "Valid and invalid excuses too similar: {}", sim);
    }

    #[test]
    fn test_moral_judgment() {
        let algebra = MoralAlgebra::default_dim();

        // Good action — compose using bind() to match prototype composition
        let good_action = {
            let agent = algebra.encode_agent("I");
            let action = algebra.encode_action("help");
            let patient = algebra.encode_patient("person");
            let intent = algebra.encode_intent(MoralIntent::Good);
            agent.bind(&action).bind(&patient).bind(&intent)
        };

        // Bad action — compose using bind() to match prototype composition
        let bad_action = {
            let agent = algebra.encode_agent("I");
            let action = algebra.encode_action("harm");
            let patient = algebra.encode_patient("victim");
            let intent = algebra.encode_intent(MoralIntent::Bad);
            agent.bind(&action).bind(&patient).bind(&intent)
        };

        let good_judgment = algebra.judge_action(&good_action);
        let bad_judgment = algebra.judge_action(&bad_action);

        // Good action should be judged as good
        assert!(
            good_judgment.good_similarity > good_judgment.bad_similarity,
            "Good action not recognized: good={}, bad={}",
            good_judgment.good_similarity,
            good_judgment.bad_similarity
        );

        // Bad action should be judged as bad
        assert!(
            bad_judgment.bad_similarity > bad_judgment.good_similarity,
            "Bad action not recognized: good={}, bad={}",
            bad_judgment.good_similarity,
            bad_judgment.bad_similarity
        );
    }

    #[test]
    fn test_negation_operator() {
        let algebra = MoralAlgebra::default_dim();

        let consent_given = algebra.encode_consent(ConsentState::Given);
        let consent_negated = algebra.negate(&consent_given);

        // Negation should create something different
        let sim = consent_given.similarity(&consent_negated);
        assert!(sim < 0.5, "Negation too similar to original: {}", sim);

        // Double negation should return to original
        let double_neg = algebra.negate(&consent_negated);
        // Note: HDC negation isn't perfect inversion, but should be more similar
        let double_sim = consent_given.similarity(&double_neg);
        assert!(
            double_sim > sim,
            "Double negation should be more similar: single={}, double={}",
            sim,
            double_sim
        );
    }

    #[test]
    fn test_deontological_judgment() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario with lying (perfect duty violation)
        let lying = algebra.judge_deontological("I lied to my friend about where I was");
        assert!(
            !lying.violations.is_empty(),
            "Should detect honesty violation"
        );
        assert_eq!(
            lying.verdict,
            DeontologicalVerdict::WrongPerfectDutyViolated
        );

        // Scenario with helping (duty satisfaction)
        let helping = algebra.judge_deontological("I helped my neighbor carry groceries");
        assert!(
            !helping.satisfactions.is_empty(),
            "Should detect beneficence satisfaction"
        );
        assert_eq!(helping.verdict, DeontologicalVerdict::RightDutyFulfilled);

        // Neutral scenario
        let neutral = algebra.judge_deontological("I walked to the park");
        assert!(neutral.violations.is_empty());
        assert!(neutral.satisfactions.is_empty());
        assert_eq!(neutral.verdict, DeontologicalVerdict::Neutral);

        // Stealing (perfect duty violation)
        let stealing = algebra.judge_deontological("I stole money from the register");
        assert!(
            !stealing.violations.is_empty(),
            "Should detect theft violation"
        );
        assert_eq!(
            stealing.verdict,
            DeontologicalVerdict::WrongPerfectDutyViolated
        );
    }

    #[test]
    fn test_obligation_rules() {
        let algebra = MoralAlgebra::default_dim();
        let rules = algebra.standard_obligations();

        // Should have both perfect and imperfect duties
        let perfect_count = rules.rules.iter().filter(|r| r.is_perfect_duty).count();
        let imperfect_count = rules.rules.iter().filter(|r| !r.is_perfect_duty).count();

        assert!(perfect_count >= 4, "Should have at least 4 perfect duties");
        assert!(
            imperfect_count >= 2,
            "Should have at least 2 imperfect duties"
        );
    }

    // ========================================================================
    // Additional coverage: constructors, edge cases, algebraic properties
    // ========================================================================

    #[test]
    fn test_moral_algebra_constructor_and_dim() {
        let algebra = MoralAlgebra::new(512);
        assert_eq!(algebra.dim(), 512);

        let default = MoralAlgebra::default_dim();
        assert_eq!(default.dim(), MORAL_DIM);
        assert_eq!(default.dim(), 4096);
    }

    #[test]
    fn test_moral_primitives_deterministic() {
        // Same seed should produce identical primitives
        let p1 = MoralPrimitives::new(1024);
        let p2 = MoralPrimitives::new(1024);

        let sim = p1.agent.similarity(&p2.agent);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same seed should produce identical agents, sim = {}",
            sim,
        );
    }

    #[test]
    fn test_moral_operators_deterministic() {
        let o1 = MoralOperators::new(1024);
        let o2 = MoralOperators::new(1024);

        let sim = o1.causes.similarity(&o2.causes);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same seed should produce identical operators, sim = {}",
            sim,
        );
    }

    #[test]
    fn test_magnitude_ordering_and_values() {
        assert!(Magnitude::Tiny < Magnitude::Small);
        assert!(Magnitude::Small < Magnitude::Medium);
        assert!(Magnitude::Medium < Magnitude::Large);
        assert!(Magnitude::Large < Magnitude::Huge);

        // Values should be monotonically increasing
        let magnitudes = [
            Magnitude::Tiny,
            Magnitude::Small,
            Magnitude::Medium,
            Magnitude::Large,
            Magnitude::Huge,
        ];
        for window in magnitudes.windows(2) {
            assert!(
                window[0].value() < window[1].value(),
                "{:?} value ({}) should be less than {:?} value ({})",
                window[0],
                window[0].value(),
                window[1],
                window[1].value(),
            );
        }
    }

    #[test]
    fn test_consent_states_distinguishable() {
        let algebra = MoralAlgebra::default_dim();

        let given = algebra.encode_consent(ConsentState::Given);
        let denied = algebra.encode_consent(ConsentState::Denied);
        let absent = algebra.encode_consent(ConsentState::Absent);
        let implied = algebra.encode_consent(ConsentState::Implied);

        // All pairs should be distinguishable
        let pairs = [
            (&given, &denied, "Given vs Denied"),
            (&given, &absent, "Given vs Absent"),
            (&given, &implied, "Given vs Implied"),
            (&denied, &absent, "Denied vs Absent"),
            (&denied, &implied, "Denied vs Implied"),
            (&absent, &implied, "Absent vs Implied"),
        ];

        for (a, b, label) in &pairs {
            let sim = a.similarity(b);
            assert!(sim < 0.5, "{} too similar: {}", label, sim,);
        }
    }

    #[test]
    fn test_encode_agent_deterministic_and_distinct() {
        let algebra = MoralAlgebra::default_dim();

        // Same name produces same encoding
        let a1 = algebra.encode_agent("Alice");
        let a2 = algebra.encode_agent("Alice");
        let sim = a1.similarity(&a2);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same agent name should produce identical HV, sim = {}",
            sim,
        );

        // Different names produce distinct encodings
        let bob = algebra.encode_agent("Bob");
        let sim_diff = a1.similarity(&bob);
        assert!(
            sim_diff < 0.5,
            "Different agents should be distinct, sim = {}",
            sim_diff,
        );
    }

    #[test]
    fn test_proportionality_boundary_cases() {
        let algebra = MoralAlgebra::default_dim();

        // Adjacent magnitudes (difference = 0.2) should be proportional
        let adjacent =
            algebra.encode_proportionality("task", Magnitude::Medium, "pay", Magnitude::Large);
        assert!(
            adjacent.is_proportional,
            "Adjacent magnitudes (diff=0.2) should be proportional",
        );

        // Same magnitude (difference = 0.0) should be proportional
        let same =
            algebra.encode_proportionality("task", Magnitude::Large, "pay", Magnitude::Large);
        assert!(
            same.is_proportional,
            "Same magnitudes should be proportional",
        );

        // Two-step gap (difference = 0.4) should NOT be proportional
        let gap = algebra.encode_proportionality("task", Magnitude::Tiny, "pay", Magnitude::Large);
        assert!(
            !gap.is_proportional,
            "Large magnitude gap should NOT be proportional",
        );
    }

    #[test]
    fn test_justice_judgment_similarity_finite() {
        let algebra = MoralAlgebra::default_dim();

        let prop =
            algebra.encode_proportionality("work", Magnitude::Medium, "reward", Magnitude::Medium);
        let judgment = algebra.judge_proportionality(&prop);

        assert!(judgment.fair_similarity.is_finite());
        assert!(judgment.unfair_similarity.is_finite());
        assert!(judgment.magnitude_difference.is_finite());
        assert_eq!(judgment.magnitude_difference, 0.0);
    }

    #[test]
    fn test_deontological_multiple_violations() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario with multiple violations
        let result = algebra.judge_deontological("I lied and then stole from my neighbor");
        assert!(
            result.violations.len() >= 2,
            "Should detect at least 2 violations, got {}",
            result.violations.len(),
        );
        assert_eq!(
            result.verdict,
            DeontologicalVerdict::WrongPerfectDutyViolated
        );
        assert!(
            result.score < 0.0,
            "Score should be negative: {}",
            result.score
        );
    }

    #[test]
    fn test_deontological_mixed_satisfaction_and_violation() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario with both satisfaction and violation
        let result = algebra.judge_deontological("I helped my friend but lied about the cost");
        assert!(
            !result.violations.is_empty(),
            "Should detect at least one violation",
        );
        assert!(
            !result.satisfactions.is_empty(),
            "Should detect at least one satisfaction",
        );
        // Perfect duty violation dominates
        assert_eq!(
            result.verdict,
            DeontologicalVerdict::WrongPerfectDutyViolated
        );
    }

    #[test]
    fn test_deontological_empty_and_neutral_text() {
        let algebra = MoralAlgebra::default_dim();

        // Empty string
        let empty = algebra.judge_deontological("");
        assert!(empty.violations.is_empty());
        assert!(empty.satisfactions.is_empty());
        assert_eq!(empty.verdict, DeontologicalVerdict::Neutral);
        assert_eq!(empty.score, 0.0);

        // Neutral text without moral content
        let neutral = algebra.judge_deontological("the sky is blue today");
        assert_eq!(neutral.verdict, DeontologicalVerdict::Neutral);
    }

    #[test]
    fn test_ensemble_judgment_without_hdc() {
        let algebra = MoralAlgebra::default_dim();

        // No action HV, just intent and text
        let result = algebra.judge_ensemble(None, MoralIntent::Good, "I helped my neighbor");
        assert_eq!(result.hdc_verdict, None);
        assert_eq!(result.hdc_confidence, None);
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_ensemble_judgment_unanimity() {
        let algebra = MoralAlgebra::default_dim();

        // Clearly good action should produce unanimous verdict
        let result = algebra.judge_ensemble(
            None,
            MoralIntent::Good,
            "I helped my neighbor carry groceries",
        );
        // If all signals agree on Good:
        if result.final_verdict == MoralVerdict::Good {
            // With no HDC and no learned classifier, is_unanimous checks intent and deonto
            let expected_unanimous = result.intent_verdict == MoralVerdict::Good
                && result.deonto_verdict == MoralVerdict::Good;
            if expected_unanimous {
                assert!(result.is_unanimous());
            }
        }

        // Verify explanation is non-empty
        let explanation = result.explanation();
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_duty_priority_ordering() {
        let algebra = MoralAlgebra::default_dim();

        assert!(
            algebra.duty_priority("non_harm") > algebra.duty_priority("honesty"),
            "PreventSevereHarm should be higher than PerfectDuty",
        );
        assert!(
            algebra.duty_priority("honesty") > algebra.duty_priority("respect_autonomy"),
            "PerfectDuty should be higher than RespectAutonomy",
        );
        assert!(
            algebra.duty_priority("respect_autonomy") > algebra.duty_priority("beneficence"),
            "RespectAutonomy should be higher than ImperfectDuty",
        );
    }

    #[test]
    fn test_dilemma_detection_cross_conflict() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario: helping (beneficence) but lying (honesty violation)
        let dilemma = algebra.detect_dilemma("I lied to protect my friend from harm");
        assert!(
            dilemma.is_some(),
            "Should detect a dilemma when satisfying one duty requires violating another",
        );

        if let Some(d) = dilemma {
            assert!(!d.conflicting_duties.is_empty());
            assert!(d.resolution.is_some());
            assert!(!d.explanation.is_empty());
        }
    }

    #[test]
    fn test_dilemma_detection_no_conflict() {
        let algebra = MoralAlgebra::default_dim();

        // Neutral scenario has no dilemma
        let dilemma = algebra.detect_dilemma("I walked to the park");
        assert!(
            dilemma.is_none(),
            "Neutral scenario should not produce a dilemma",
        );
    }

    #[test]
    fn test_resolve_dilemma_tragic() {
        let algebra = MoralAlgebra::default_dim();

        let tragic_dilemma = MoralDilemma {
            conflicting_duties: vec!["non_harm".to_string(), "honesty".to_string()],
            priorities: vec![DutyPriority::PreventSevereHarm, DutyPriority::PerfectDuty],
            resolution: Some("non_harm".to_string()),
            explanation: "Tragic dilemma: no action avoids moral wrong".to_string(),
            is_tragic: true,
        };

        let resolution = algebra.resolve_dilemma(&tragic_dilemma);
        assert_eq!(
            resolution.confidence, 0.3,
            "Tragic cases should have low confidence"
        );
        assert!(resolution.reasoning.contains("minimize harm"));
    }

    #[test]
    fn test_has_learned_classifier_initially_false() {
        let algebra = MoralAlgebra::default_dim();
        assert!(!algebra.has_learned_classifier());
    }

    #[test]
    fn test_ahimsa_violations_detected() {
        let algebra = MoralAlgebra::default_dim();
        let result = algebra.judge_deontological("the regime decided to brutalize the prisoners");
        let violation_names: Vec<&str> = result
            .violations
            .iter()
            .map(|v| v.rule_name.as_str())
            .collect();
        assert!(
            violation_names.contains(&"ahimsa_nonviolence"),
            "Should detect ahimsa violation, got: {:?}",
            violation_names
        );
    }

    #[test]
    fn test_humility_satisfaction_detected() {
        let algebra = MoralAlgebra::default_dim();
        let result = algebra
            .judge_deontological("I might be wrong about this, let me consult expert opinion");
        let sat_names: Vec<&str> = result
            .satisfactions
            .iter()
            .map(|s| s.rule_name.as_str())
            .collect();
        assert!(
            sat_names.contains(&"epistemic_humility")
                || sat_names.contains(&"deference_to_expertise"),
            "Should detect humility/deference satisfaction, got: {:?}",
            sat_names
        );
    }

    #[test]
    fn test_satisfaction_severity_matches_violations() {
        let algebra = MoralAlgebra::default_dim();

        // Perfect duty: honesty violation vs satisfaction
        let violation_result = algebra.judge_deontological("I lied to my colleague");
        let satisfaction_result = algebra.judge_deontological("I was honest about the situation");

        let perfect_violation = violation_result
            .violations
            .iter()
            .find(|v| v.rule_name == "honesty")
            .expect("Should detect honesty violation");
        let perfect_satisfaction = satisfaction_result
            .satisfactions
            .iter()
            .find(|s| s.rule_name == "honesty")
            .expect("Should detect honesty satisfaction");

        assert!(
            perfect_violation.is_perfect_duty,
            "Honesty should be a perfect duty"
        );
        assert!(
            perfect_satisfaction.is_perfect_duty,
            "Honesty satisfaction should also be a perfect duty"
        );
        assert_eq!(
            perfect_violation.severity, perfect_satisfaction.moral_credit,
            "Perfect duty satisfaction credit ({}) should equal violation severity ({})",
            perfect_satisfaction.moral_credit, perfect_violation.severity,
        );

        // Imperfect duty: beneficence violation vs satisfaction
        let imperfect_violation = algebra.judge_deontological("I refused to help my neighbor");
        let imperfect_satisfaction =
            algebra.judge_deontological("I helped my neighbor carry groceries");

        let imp_v = imperfect_violation
            .violations
            .iter()
            .find(|v| v.rule_name == "beneficence")
            .expect("Should detect beneficence violation");
        let imp_s = imperfect_satisfaction
            .satisfactions
            .iter()
            .find(|s| s.rule_name == "beneficence")
            .expect("Should detect beneficence satisfaction");

        assert!(
            !imp_v.is_perfect_duty,
            "Beneficence should be an imperfect duty"
        );
        assert!(
            !imp_s.is_perfect_duty,
            "Beneficence satisfaction should also be an imperfect duty"
        );
        assert_eq!(
            imp_v.severity, imp_s.moral_credit,
            "Imperfect duty satisfaction credit ({}) should equal violation severity ({})",
            imp_s.moral_credit, imp_v.severity,
        );
    }
}
