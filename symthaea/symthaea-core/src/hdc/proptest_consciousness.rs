//! Property-based tests for consciousness modules
//!
//! Tests invariants of SemanticBridge, ConsciousnessVerifier, and PhiFeedbackController
//! using proptest with reduced case counts for speed.

use proptest::prelude::*;

use super::semantic_bridge::SemanticBridge;
use super::consciousness_verifier::{ConsciousnessVerifier, ConsciousnessVerdict};
use super::phi_feedback::{PhiFeedbackController, PhiFeedbackConfig};
use super::consciousness_integration::ConsciousnessState;

// =============================================================================
// STRATEGIES
// =============================================================================

/// Generate a word from a fixed vocabulary (ensures it's known to the bridge)
fn known_word() -> impl Strategy<Value = String> {
    prop::sample::select(vec![
        "cat", "dog", "bird", "fish", "tree",
        "sky", "sun", "moon", "star", "rain",
        "red", "blue", "green", "big", "small",
    ]).prop_map(|s| s.to_string())
}

/// Generate a Phi value in [0, 1]
fn phi_value() -> impl Strategy<Value = f64> {
    (0u32..=1000).prop_map(|n| n as f64 / 1000.0)
}

/// Generate a sequence of Phi values
fn phi_sequence(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(phi_value(), min_len..=max_len)
}

// =============================================================================
// SEMANTIC BRIDGE PROPERTIES
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Same word always encodes to the same HV (determinism).
    #[test]
    fn prop_encode_is_deterministic(word in known_word()) {
        let bridge = SemanticBridge::with_vocabulary(&[
            "cat", "dog", "bird", "fish", "tree",
            "sky", "sun", "moon", "star", "rain",
            "red", "blue", "green", "big", "small",
        ]);
        let hv1 = bridge.encode_word(&word);
        let hv2 = bridge.encode_word(&word);
        prop_assert_eq!(hv1, hv2, "Same word should always produce same HV");
    }

    /// Encode a word, then decode_hv should recover it as the top-1 result.
    #[test]
    fn prop_encode_decode_roundtrip(word in known_word()) {
        let bridge = SemanticBridge::with_vocabulary(&[
            "cat", "dog", "bird", "fish", "tree",
            "sky", "sun", "moon", "star", "rain",
            "red", "blue", "green", "big", "small",
        ]);
        let hv = bridge.encode_word(&word).unwrap();
        let decoded = bridge.decode_hv(&hv, 1);
        prop_assert!(!decoded.is_empty(), "decode_hv should return at least one result");
        prop_assert_eq!(&decoded[0].0, &word,
            "Top-1 decoded word should match original: got {:?}", decoded[0].0);
    }

    /// "A B" and "B A" produce different HVs (order sensitivity via permutation).
    #[test]
    fn prop_phrase_order_sensitive(
        word_a in known_word(),
        word_b in known_word().prop_filter("must be different", |b| b != "cat")
    ) {
        // Only test when words are actually different
        prop_assume!(word_a != word_b);

        let bridge = SemanticBridge::with_vocabulary(&[
            "cat", "dog", "bird", "fish", "tree",
            "sky", "sun", "moon", "star", "rain",
            "red", "blue", "green", "big", "small",
        ]);
        let phrase_ab = format!("{} {}", word_a, word_b);
        let phrase_ba = format!("{} {}", word_b, word_a);
        let hv_ab = bridge.encode_phrase(&phrase_ab);
        let hv_ba = bridge.encode_phrase(&phrase_ba);

        // They should be different (not identical) due to positional permutation
        let sim = hv_ab.similarity(&hv_ba);
        prop_assert!(sim < 1.0, "Swapped order should produce different HV, got sim={}", sim);
    }

    /// Adding the same word twice doesn't change the encoding.
    #[test]
    fn prop_vocabulary_addition_idempotent(word in known_word()) {
        let mut bridge = SemanticBridge::with_vocabulary(&["placeholder"]);
        bridge.add_word(&word);
        let hv1 = bridge.encode_word(&word).unwrap();
        bridge.add_word(&word);  // Add again
        let hv2 = bridge.encode_word(&word).unwrap();
        prop_assert_eq!(hv1, hv2, "Adding same word twice should be idempotent");
    }

    /// Semantic similarity is symmetric: sim(a, b) == sim(b, a).
    #[test]
    fn prop_semantic_similarity_symmetric(word_a in known_word(), word_b in known_word()) {
        let bridge = SemanticBridge::with_vocabulary(&[
            "cat", "dog", "bird", "fish", "tree",
            "sky", "sun", "moon", "star", "rain",
            "red", "blue", "green", "big", "small",
        ]);
        let sim_ab = bridge.semantic_similarity(&word_a, &word_b);
        let sim_ba = bridge.semantic_similarity(&word_b, &word_a);
        prop_assert_eq!(sim_ab, sim_ba, "Similarity should be symmetric");
    }
}

// =============================================================================
// CONSCIOUSNESS VERIFIER PROPERTIES
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Higher Phi → higher or equal verdict (monotonicity).
    #[test]
    fn prop_verdict_monotonic_in_phi(phi_low in 0u32..500, phi_high in 500u32..1001) {
        let phi_low = phi_low as f64 / 1000.0;
        let phi_high = phi_high as f64 / 1000.0;
        prop_assume!(phi_high >= phi_low);

        let verdict_low = ConsciousnessVerifier::verdict_from_phi(phi_low);
        let verdict_high = ConsciousnessVerifier::verdict_from_phi(phi_high);

        let rank = |v: ConsciousnessVerdict| -> u8 {
            match v {
                ConsciousnessVerdict::NotConscious => 0,
                ConsciousnessVerdict::LikelyNotConscious => 1,
                ConsciousnessVerdict::Indeterminate => 2,
                ConsciousnessVerdict::LikelyConscious => 3,
                ConsciousnessVerdict::StronglyConscious => 4,
            }
        };

        prop_assert!(rank(verdict_high) >= rank(verdict_low),
            "Higher phi ({}) should give >= verdict than lower phi ({}): {:?} vs {:?}",
            phi_high, phi_low, verdict_high, verdict_low);
    }

    /// Same input state → same verification report (determinism).
    #[test]
    fn prop_verification_deterministic(phi in phi_value()) {
        let verifier = ConsciousnessVerifier::new();
        let mut state = ConsciousnessState::default();
        state.phi = phi;
        state.consciousness_level = phi * 0.7;

        let report1 = verifier.verify_from_state(&state);
        let report2 = verifier.verify_from_state(&state);

        prop_assert!((report1.consensus_phi - report2.consensus_phi).abs() < 1e-10,
            "Same state should give same consensus_phi");
        prop_assert_eq!(report1.verdict, report2.verdict,
            "Same state should give same verdict");
    }

    /// Confidence is always in [0, 1].
    #[test]
    fn prop_confidence_bounded(phi in phi_value()) {
        let verifier = ConsciousnessVerifier::new();
        let mut state = ConsciousnessState::default();
        state.phi = phi;
        state.consciousness_level = phi;

        let report = verifier.verify_from_state(&state);
        prop_assert!(report.confidence >= 0.0 && report.confidence <= 1.0,
            "Confidence should be in [0,1], got {}", report.confidence);
    }

    /// All IIT axiom scores are in [0, 1].
    #[test]
    fn prop_axiom_scores_bounded(phi in phi_value()) {
        let verifier = ConsciousnessVerifier::new();
        let mut state = ConsciousnessState::default();
        state.phi = phi;
        state.consciousness_level = phi * 0.8;

        let report = verifier.verify_from_state(&state);
        let scores = &report.iit_axiom_scores;

        prop_assert!(scores.information >= 0.0 && scores.information <= 1.0,
            "information score out of bounds: {}", scores.information);
        prop_assert!(scores.integration >= 0.0 && scores.integration <= 1.0,
            "integration score out of bounds: {}", scores.integration);
        prop_assert!(scores.exclusion >= 0.0 && scores.exclusion <= 1.0,
            "exclusion score out of bounds: {}", scores.exclusion);
        prop_assert!(scores.composition >= 0.0 && scores.composition <= 1.0,
            "composition score out of bounds: {}", scores.composition);
        prop_assert!(scores.intrinsicality >= 0.0 && scores.intrinsicality <= 1.0,
            "intrinsicality score out of bounds: {}", scores.intrinsicality);
    }
}

// =============================================================================
// PHI FEEDBACK CONTROLLER PROPERTIES
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Phi EMA stays in [0, 1] after any sequence of steps.
    #[test]
    fn prop_ema_bounded(phis in phi_sequence(1, 50)) {
        let mut ctrl = PhiFeedbackController::new();
        for phi in &phis {
            ctrl.step(*phi);
        }
        prop_assert!(ctrl.phi_ema() >= 0.0 && ctrl.phi_ema() <= 1.0,
            "EMA should be in [0,1], got {}", ctrl.phi_ema());
    }

    /// Step count equals number of step() calls.
    #[test]
    fn prop_step_count_equals_calls(n in 1usize..100) {
        let mut ctrl = PhiFeedbackController::new();
        for i in 0..n {
            ctrl.step(0.5);
        }
        prop_assert_eq!(ctrl.step_count(), n as u64,
            "step_count should equal number of calls");
    }

    /// Modulation fields (binding_threshold_delta, workspace_capacity_delta) stay reasonable.
    #[test]
    fn prop_modulation_fields_bounded(phis in phi_sequence(1, 30)) {
        let mut ctrl = PhiFeedbackController::new();
        for phi in &phis {
            let modulation = ctrl.step(*phi);
            prop_assert!(modulation.binding_threshold >= 0.0 && modulation.binding_threshold <= 1.0,
                "binding_threshold out of bounds: {}", modulation.binding_threshold);
            prop_assert!(modulation.workspace_capacity >= 1 && modulation.workspace_capacity <= 20,
                "workspace_capacity out of bounds: {}", modulation.workspace_capacity);
            prop_assert!(modulation.consciousness_threshold >= 0.0 && modulation.consciousness_threshold <= 1.0,
                "consciousness_threshold out of bounds: {}", modulation.consciousness_threshold);
            prop_assert!(modulation.attention_gain > 0.0,
                "attention_gain should be positive: {}", modulation.attention_gain);
        }
    }

    /// After reset(), phi_history is empty and step_count is 0.
    #[test]
    fn prop_reset_clears_history(phis in phi_sequence(1, 20)) {
        let mut ctrl = PhiFeedbackController::new();
        for phi in &phis {
            ctrl.step(*phi);
        }
        prop_assert!(ctrl.step_count() > 0);

        ctrl.reset();
        prop_assert_eq!(ctrl.step_count(), 0, "step_count should be 0 after reset");
        prop_assert!(ctrl.phi_history().is_empty(), "phi_history should be empty after reset");
        prop_assert!((ctrl.phi_ema() - 0.0).abs() < 1e-10, "phi_ema should be 0 after reset");
    }
}
