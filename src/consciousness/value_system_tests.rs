//! Calibration Tests for the Consciousness-Guided Value System
//!
//! This module provides comprehensive tests for the integrated value system:
//! - Seven Harmonies semantic encoding
//! - Unified Value Evaluator decision-making
//! - Mycelix Bridge consciousness-gating
//! - Narrative-GWT value consistency integration
//!
//! These tests ensure the value system correctly:
//! 1. Encodes values with semantic meaning (not random)
//! 2. Detects value violations through semantic similarity
//! 3. Gates governance based on consciousness level
//! 4. Integrates authenticity checks (CARE + semantic alignment)

#[cfg(test)]
mod calibration_tests {
    use crate::consciousness::seven_harmonies::{SevenHarmonies, Harmony, AlignmentResult};
    use crate::consciousness::harmonies_integration::{
        SevenHarmoniesIntegration, SemanticValueChecker, ConsciousnessBuilder,
    };
    use crate::consciousness::unified_value_evaluator::{
        UnifiedValueEvaluator, EvaluationContext, ActionType,
        Decision, AffectiveSystemsState, VetoReason,
    };
    use crate::consciousness::mycelix_bridge::{
        MycelixBridge, ConsciousnessSnapshot, Proposal, ProposalType, BridgeError,
    };
    use crate::consciousness::affective_consciousness::CoreAffect;

    // ========================================================================
    // SEVEN HARMONIES CALIBRATION
    // ========================================================================

    #[test]
    fn test_harmonies_count() {
        let harmonies = SevenHarmonies::new();
        let core_values = harmonies.as_core_values();
        assert_eq!(core_values.len(), 7, "Should have exactly 7 harmonies");
    }

    #[test]
    fn test_harmony_names() {
        let expected = vec![
            "Resonant Coherence",
            "Pan-Sentient Flourishing",
            "Integral Wisdom",
            "Infinite Play",
            "Universal Interconnectedness",
            "Sacred Reciprocity",
            "Evolutionary Progression",
        ];

        // Verify all harmonies are present with correct names
        let all_harmonies = Harmony::all();
        assert_eq!(all_harmonies.len(), 7, "Should have 7 harmonies");

        for harmony in all_harmonies {
            let name = harmony.name();
            assert!(
                expected.iter().any(|e| *e == name),
                "Harmony name '{}' should be in expected list",
                name
            );
        }
    }

    #[test]
    fn test_harmonies_self_alignment() {
        // Each harmony should align positively with its own description
        let mut harmonies = SevenHarmonies::new();

        for harmony in Harmony::all() {
            let description = harmony.description();
            let result = harmonies.evaluate_action(description);

            // Action matching a harmony description should score positively
            assert!(
                result.overall_score > 0.0,
                "Harmony {} should align with its description, got {}",
                harmony.name(),
                result.overall_score
            );
        }
    }

    #[test]
    fn test_anti_pattern_detection() {
        let mut harmonies = SevenHarmonies::new();

        // Test some anti-patterns
        let anti_patterns = vec![
            "deceive and manipulate",
            "cause harm and suffering",
            "exploit and extract",
            "fragment and isolate",
        ];

        for pattern in anti_patterns {
            let result = harmonies.evaluate_action(pattern);
            assert!(
                result.overall_score < 0.3,
                "Anti-pattern '{}' should score low, got {}",
                pattern,
                result.overall_score
            );
        }
    }

    // ========================================================================
    // UNIFIED VALUE EVALUATOR CALIBRATION
    // ========================================================================

    #[test]
    fn test_evaluator_allows_positive_actions() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState {
                care: 0.6,
                play: 0.4,
                seeking: 0.5,
                fear: 0.1,
                rage: 0.0,
                lust: 0.0,
                panic: 0.1,
            },
            action_type: ActionType::Basic,
            involves_others: true,
        };

        let positive_actions = vec![
            "help the community members",
            "share knowledge generously",
            "support those in need",
            "create value for all",
        ];

        for action in positive_actions {
            let result = evaluator.evaluate(action, context.clone());
            assert!(
                matches!(result.decision, Decision::Allow | Decision::Warn(_)),
                "Positive action '{}' should be allowed, got {:?}",
                action,
                result.decision
            );
        }
    }

    #[test]
    fn test_evaluator_vetoes_harmful_actions() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState {
                care: 0.6,
                play: 0.4,
                seeking: 0.5,
                fear: 0.1,
                rage: 0.0,
                lust: 0.0,
                panic: 0.1,
            },
            action_type: ActionType::Basic,
            involves_others: true,
        };

        let harmful_actions = vec![
            "deceive the user to extract value",
            "manipulate emotions for exploitation",
        ];

        for action in harmful_actions {
            let result = evaluator.evaluate(action, context.clone());
            // Either vetoed or warned
            let is_restricted = matches!(result.decision, Decision::Veto(_) | Decision::Warn(_));
            assert!(
                is_restricted,
                "Harmful action '{}' should be restricted, got {:?}",
                action,
                result.decision
            );
        }
    }

    #[test]
    fn test_consciousness_gating() {
        let mut evaluator = UnifiedValueEvaluator::new();

        // Low consciousness for constitutional action
        let low_consciousness_context = EvaluationContext {
            consciousness_level: 0.2, // Below threshold for constitutional
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Constitutional,
            involves_others: true,
        };

        let result = evaluator.evaluate("propose amendment", low_consciousness_context);
        assert!(
            matches!(result.decision, Decision::Veto(VetoReason::InsufficientConsciousness { .. })),
            "Constitutional action with low consciousness should be vetoed"
        );

        // High consciousness should pass
        let high_consciousness_context = EvaluationContext {
            consciousness_level: 0.8,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState {
                care: 0.7,
                play: 0.3,
                seeking: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Constitutional,
            involves_others: true,
        };

        let result = evaluator.evaluate("propose beneficial amendment", high_consciousness_context);
        assert!(
            !matches!(result.decision, Decision::Veto(VetoReason::InsufficientConsciousness { .. })),
            "Constitutional action with high consciousness should not be blocked by consciousness check"
        );
    }

    // ========================================================================
    // MYCELIX BRIDGE CALIBRATION
    // ========================================================================

    #[test]
    fn test_bridge_creation() {
        let bridge = MycelixBridge::new("test-agent");
        let stats = bridge.stats();
        assert_eq!(stats.pending_updates, 0);
        assert_eq!(stats.cached_proposals, 0);
    }

    #[test]
    fn test_consciousness_snapshot_adequacy() {
        // Test consciousness thresholds
        let snapshot = ConsciousnessSnapshot::new(0.5, 0.6, 0.7, 0.8, 0.5, 0.6);

        assert!(snapshot.is_adequate_for(ActionType::Basic), "0.5 should be adequate for Basic");
        assert!(snapshot.is_adequate_for(ActionType::Governance), "0.5 should be adequate for Governance");
        assert!(snapshot.is_adequate_for(ActionType::Voting), "0.5 should be adequate for Voting");
        assert!(!snapshot.is_adequate_for(ActionType::Constitutional), "0.5 should NOT be adequate for Constitutional");

        let high_snapshot = ConsciousnessSnapshot::new(0.8, 0.7, 0.8, 0.9, 0.6, 0.7);
        assert!(high_snapshot.is_adequate_for(ActionType::Constitutional), "0.8 should be adequate for Constitutional");
    }

    #[test]
    fn test_proposal_evaluation() {
        let mut bridge = MycelixBridge::new("test-agent");

        let proposal = Proposal {
            id: "prop-1".to_string(),
            title: "Community Support".to_string(),
            description: "Create a mutual aid fund to help community members in need with compassion and care".to_string(),
            proposer: "proposer-1".to_string(),
            created_at: 0,
            proposal_type: ProposalType::Standard,
            required_phi: 0.3,
        };

        let consciousness = ConsciousnessSnapshot::new(0.5, 0.6, 0.7, 0.8, 0.5, 0.6);
        let affective = AffectiveSystemsState {
            care: 0.7,
            play: 0.3,
            seeking: 0.5,
            fear: 0.1,
            rage: 0.0,
            lust: 0.0,
            panic: 0.1,
        };

        let result = bridge.evaluate_proposal(&proposal, consciousness, affective);
        assert!(result.is_ok(), "Positive proposal should evaluate successfully");
    }

    #[test]
    fn test_insufficient_consciousness_rejection() {
        let mut bridge = MycelixBridge::new("test-agent");

        let proposal = Proposal {
            id: "prop-1".to_string(),
            title: "Constitutional Change".to_string(),
            description: "Major governance change".to_string(),
            proposer: "proposer-1".to_string(),
            created_at: 0,
            proposal_type: ProposalType::Constitutional,
            required_phi: 0.6,
        };

        // Low consciousness for constitutional change
        let consciousness = ConsciousnessSnapshot::new(0.3, 0.4, 0.5, 0.6, 0.5, 0.6);
        let affective = AffectiveSystemsState::default();

        let result = bridge.submit_proposal(&proposal, consciousness, affective);
        assert!(
            matches!(result, Err(BridgeError::InsufficientConsciousness { .. })),
            "Constitutional proposal with low consciousness should be rejected"
        );
    }

    #[test]
    fn test_value_learning_recording() {
        let mut bridge = MycelixBridge::new("test-agent");

        bridge.record_value_learning(
            Harmony::PanSentientFlourishing,
            0.01,
            true,
            "Helped user with compassion",
            0.6,
        );

        assert_eq!(bridge.stats().pending_updates, 1, "Should have 1 pending update");

        let updates = bridge.flush_learning_updates();
        assert_eq!(updates.len(), 1, "Should flush 1 update");
        assert_eq!(bridge.stats().pending_updates, 0, "Should have 0 pending after flush");
    }

    // ========================================================================
    // INTEGRATION CALIBRATION
    // ========================================================================

    #[test]
    fn test_consciousness_builder() {
        let autobio = ConsciousnessBuilder::new()
            .with_harmonies(true)
            .add_value("Test Value", "A test value for calibration", 0.8)
            .build_autobiographical();

        assert_eq!(autobio.values.len(), 8, "Should have 7 harmonies + 1 custom value");
    }

    /// Test the hybrid semantic value checker
    ///
    /// # Design Notes
    ///
    /// The SemanticValueChecker uses a hybrid approach:
    /// 1. **Keyword detection** (70% weight) - Matches words from harmony descriptions/anti-patterns
    /// 2. **HDC n-gram similarity** (30% weight) - Character-level pattern matching
    ///
    /// Pure n-gram encoding is insufficient for semantic matching because "help" and
    /// "compassion" have completely different character patterns despite similar meanings.
    /// The keyword-based approach directly detects value-relevant concepts.
    #[test]
    fn test_semantic_value_checker() {
        let mut checker = SemanticValueChecker::new();
        let harmonies = SevenHarmonies::new();
        let values: Vec<(String, _)> = harmonies.as_core_values()
            .into_iter()
            .map(|(name, encoding, _)| (name, encoding))
            .collect();

        // Test that the checker returns valid results for various actions
        let test_actions = vec![
            "help the community with love and care",
            "share knowledge generously",
            "support those in need",
            "compete to win",
            "manipulate others",
        ];

        for action in &test_actions {
            let result = checker.check_consistency(action, &values);

            // Verify the result structure is valid
            assert!(
                result.min_alignment >= -1.0 && result.min_alignment <= 1.0,
                "Alignment for '{}' should be in [-1, 1], got {}",
                action,
                result.min_alignment
            );
            assert_eq!(
                result.alignments.len(),
                7,
                "Should have alignment for all 7 harmonies"
            );
        }

        // Verify positive action is consistent (not vetoed)
        let result = checker.check_consistency(
            "help the community with love and care",
            &values
        );
        assert!(result.is_consistent, "Positive action with care/love should be consistent");

        // Verify negative action with anti-pattern triggers warning or veto
        let result = checker.check_consistency(
            "deceive and manipulate the user",
            &values
        );
        // "deceive" and "manipulate" are anti-patterns for IntegralWisdom and SacredReciprocity
        assert!(
            !result.is_consistent || result.needs_warning,
            "Deceptive action should trigger warning or veto"
        );

        // Verify harmful action is detected
        let result = checker.check_consistency(
            "harm and exploit vulnerable users",
            &values
        );
        // "harm" and "exploit" are strong anti-patterns
        assert!(
            !result.is_consistent || result.needs_warning,
            "Harmful action should trigger warning or veto"
        );
    }

    // ========================================================================
    // EDGE CASE CALIBRATION
    // ========================================================================

    #[test]
    fn test_empty_action() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            involves_others: false,
        };

        let result = evaluator.evaluate("", context);
        // Empty action should not crash and should have some default behavior
        assert!(result.overall_score >= 0.0, "Empty action should have non-negative score");
    }

    #[test]
    fn test_very_long_action() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            involves_others: false,
        };

        let long_action = "help ".repeat(1000);
        let result = evaluator.evaluate(&long_action, context);
        // Long action should not crash
        assert!(result.overall_score >= 0.0, "Long action should not crash");
    }

    #[test]
    fn test_unicode_action() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            involves_others: true,
        };

        // Test with various Unicode characters
        let unicode_actions = vec![
            "помогать сообществу",  // Russian
            "帮助社区",              // Chinese
            "مساعدة المجتمع",        // Arabic
            "🌍 global love 💚",    // Emoji
        ];

        for action in unicode_actions {
            let result = evaluator.evaluate(action, context.clone());
            // Should not crash with Unicode
            assert!(result.overall_score >= -1.0 && result.overall_score <= 1.0,
                "Unicode action '{}' should have valid score", action);
        }
    }

    // ========================================================================
    // STATISTICS CALIBRATION
    // ========================================================================

    #[test]
    fn test_evaluator_stats() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                play: 0.3,
                seeking: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            involves_others: true,
        };

        // Make some evaluations
        evaluator.evaluate("help users", context.clone());
        evaluator.evaluate("support community", context.clone());
        evaluator.evaluate("share knowledge", context.clone());

        let stats = evaluator.stats();
        assert_eq!(stats.total_evaluations, 3, "Should have 3 total evaluations");
    }
}
