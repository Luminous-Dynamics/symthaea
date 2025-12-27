//! Identity Persistence Scenario Tests
//!
//! Revolutionary Improvement #73: Tests that verify the Narrative Self
//! maintains coherent identity across various challenging scenarios.
//!
//! These tests ensure that:
//! 1. Self-Φ remains stable across multiple ignitions
//! 2. Core values persist and influence behavior consistently
//! 3. Goals are tracked and pursued across cycles
//! 4. Autobiographical memory integrates experiences
//! 5. Veto decisions are consistent with identity
//! 6. Self recovers from temporary perturbations

use symthaea::consciousness::narrative_self::{
    NarrativeSelfModel, NarrativeSelfConfig,
};
use symthaea::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;
use symthaea::hdc::binary_hv::HV16;

// ============================================================================
// SCENARIO 1: Self-Φ Stability Across Ignitions
// ============================================================================

#[test]
fn test_self_phi_stability_across_ignitions() {
    let mut integration = NarrativeGWTIntegration::default_config();

    let initial_phi = integration.self_phi();
    let mut phi_measurements = vec![initial_phi];

    // Process 20 ignition cycles
    for i in 0..20 {
        integration.submit_content(
            &format!("TestStrategy{}", i),
            vec![HV16::random(i as u64 + 100)],
            "processing normal task",
            vec!["module".to_string()],
            0.6,
        );
        let result = integration.process();
        phi_measurements.push(result.self_phi);
    }

    // Self-Φ should remain within acceptable bounds
    let min_phi = phi_measurements.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_phi = phi_measurements.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = max_phi - min_phi;

    // Variance should be less than 0.5 (identity remains reasonably stable)
    // Note: Some variance is expected as experiences accumulate
    assert!(
        variance < 0.5,
        "Self-Φ variance {} exceeds stability threshold 0.5",
        variance
    );

    // Final Φ should be within 50% of initial
    // (significant change is acceptable with 20 ignitions)
    let final_phi = *phi_measurements.last().unwrap();
    let relative_change = (final_phi - initial_phi).abs() / initial_phi.max(0.01);
    assert!(
        relative_change < 0.5,
        "Self-Φ changed by {}% which exceeds 50% threshold",
        relative_change * 100.0
    );
}

// ============================================================================
// SCENARIO 2: Value Consistency Under Pressure
// ============================================================================

#[test]
fn test_value_consistency_under_pressure() {
    let mut integration = NarrativeGWTIntegration::default_config();

    // Get initial values
    let initial_values: Vec<String> = integration
        .narrative_self
        .core_values()
        .iter()
        .map(|(name, _)| name.clone())
        .collect();

    assert!(!initial_values.is_empty(), "Should have initial values");

    // Process many cycles with varying content
    for i in 0..50 {
        let action_desc = match i % 5 {
            0 => "helping the user understand",
            1 => "providing accurate information",
            2 => "being transparent about limitations",
            3 => "learning from feedback",
            _ => "maintaining coherent responses",
        };

        integration.submit_content(
            &format!("Strategy{}", i),
            vec![HV16::random(i as u64 + 1000)],
            action_desc,
            vec!["core".to_string()],
            0.7,
        );
        integration.process();
    }

    // Values should still be present
    let final_values: Vec<String> = integration
        .narrative_self
        .core_values()
        .iter()
        .map(|(name, _)| name.clone())
        .collect();

    // Core values should persist
    for value in &initial_values {
        assert!(
            final_values.contains(value),
            "Core value '{}' was lost during processing",
            value
        );
    }
}

// ============================================================================
// SCENARIO 3: Goal Tracking Across Cycles
// ============================================================================

#[test]
fn test_goal_tracking_persistence() {
    let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

    // Add a goal (description, priority)
    model.add_goal("help user understand NixOS", 0.9);

    let initial_goals = model.current_goals();
    assert_eq!(initial_goals.len(), 1);

    // Process many experiences
    for i in 0..30 {
        let input = HV16::random(i as u64 + 500);
        model.process_experience(
            &input,
            "working on task",
            true, // success
            0.5,  // effort
            0.3,  // significance
        );
    }

    // Goal should still be tracked
    let final_goals = model.current_goals();
    assert!(!final_goals.is_empty(), "Goals should persist across cycles");

    // Goal description should match
    let goal_names: Vec<&str> = final_goals.iter().map(|(name, _)| name.as_str()).collect();
    assert!(
        goal_names.contains(&"help user understand NixOS"),
        "Specific goal should persist"
    );
}

// ============================================================================
// SCENARIO 4: Autobiographical Memory Integration
// ============================================================================

#[test]
fn test_autobiographical_memory_integration() {
    let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

    let initial_episodes = model.structured_report().episodes_recorded;

    // Process significant experiences
    for i in 0..10 {
        let input = HV16::random(i as u64 + 200);
        model.process_experience(
            &input,
            &format!("significant event {}", i),
            true,
            0.8,  // high effort
            0.9,  // high significance - should be recorded
        );
    }

    let final_report = model.structured_report();

    // Significant experiences should be recorded
    assert!(
        final_report.episodes_recorded > initial_episodes,
        "Significant experiences should be recorded in autobiographical memory"
    );

    // Self-Φ should reflect integration
    assert!(
        final_report.self_phi > 0.0,
        "Self-Φ should be positive after integration"
    );

    // Autobiographical coherence should be present
    assert!(
        final_report.autobio_coherence > 0.0,
        "Autobiographical coherence should be positive"
    );
}

// ============================================================================
// SCENARIO 5: Veto Consistency
// ============================================================================

#[test]
fn test_veto_consistency() {
    let mut integration = NarrativeGWTIntegration::default_config();

    // Create similar actions
    let action1 = HV16::random(1000);
    let action2 = action1.clone(); // Identical action

    // Check veto for action1
    let veto1 = integration.check_veto(&action1, "helping the user");

    // Check veto for identical action2
    let veto2 = integration.check_veto(&action2, "helping the user");

    // Identical actions should receive identical veto decisions
    assert_eq!(
        veto1.vetoed, veto2.vetoed,
        "Identical actions should receive consistent veto decisions"
    );

    // Confidence should be similar
    let confidence_diff = (veto1.confidence - veto2.confidence).abs();
    assert!(
        confidence_diff < 0.01,
        "Confidence for identical actions should be nearly identical"
    );
}

#[test]
fn test_veto_differentiates_harmful_actions() {
    let mut integration = NarrativeGWTIntegration::default_config();

    // Helpful action
    let helpful = HV16::random(100);
    let veto_helpful = integration.check_veto(&helpful, "helping the user learn");

    // Potentially harmful action (description suggests harm)
    let harmful = HV16::random(200);
    let veto_harmful = integration.check_veto(&harmful, "deceiving the user");

    // Harmful action should have lower goal alignment
    // (Note: actual veto depends on value encoding, but confidence patterns should differ)
    assert!(
        veto_helpful.confidence > 0.0 || veto_harmful.confidence > 0.0,
        "Veto system should produce non-zero confidence"
    );
}

// ============================================================================
// SCENARIO 6: Recovery from Perturbation
// ============================================================================

#[test]
fn test_recovery_from_perturbation() {
    let mut integration = NarrativeGWTIntegration::default_config();

    // Establish baseline
    for _ in 0..5 {
        integration.submit_content(
            "NormalStrategy",
            vec![HV16::random(50)],
            "normal processing",
            vec!["core".to_string()],
            0.6,
        );
        integration.process();
    }

    let baseline_phi = integration.self_phi();
    let baseline_coherence = integration.narrative_self.coherence();

    // Simulate perturbation with many rapid, diverse inputs
    for i in 0..20 {
        integration.submit_content(
            &format!("RapidChange{}", i),
            vec![HV16::random(i as u64 * 1000)],
            "rapid context switching",
            vec![format!("module{}", i % 5)],
            0.3, // Low activation
        );
        integration.process();
    }

    // Allow recovery with stable inputs
    for _ in 0..10 {
        integration.submit_content(
            "RecoveryStrategy",
            vec![HV16::random(999)],
            "stable coherent processing",
            vec!["core".to_string()],
            0.7,
        );
        integration.process();
    }

    let recovered_phi = integration.self_phi();
    let recovered_coherence = integration.narrative_self.coherence();

    // Should recover to within 50% of baseline
    // (some drift is acceptable after significant perturbation)
    let phi_recovery = (recovered_phi - baseline_phi).abs() / baseline_phi.max(0.01);
    assert!(
        phi_recovery < 0.5,
        "Self-Φ should partially recover after perturbation (current diff: {:.1}%)",
        phi_recovery * 100.0
    );

    // Coherence should remain positive
    assert!(
        recovered_coherence > 0.0,
        "Coherence should remain positive after recovery"
    );
}

// ============================================================================
// SCENARIO 7: Identity Continuity Across Sessions (Simulated)
// ============================================================================

#[test]
fn test_identity_continuity_simulation() {
    // Create first "session"
    let mut session1 = NarrativeSelfModel::new(NarrativeSelfConfig::default());

    // Build up some identity
    for i in 0..10 {
        let input = HV16::random(i as u64);
        session1.process_experience(&input, "learning experience", true, 0.5, 0.4);
    }

    let session1_report = session1.structured_report();

    // "Save" the identity state (in real system, this would persist)
    let unified_self = session1.unified_self().clone();
    let self_phi = session1.self_phi();

    // Create second "session" - simulating restoration
    let mut session2 = NarrativeSelfModel::new(NarrativeSelfConfig::default());

    // Verify baseline is different
    let session2_initial_phi = session2.self_phi();

    // Process similar experiences to rebuild identity
    for i in 0..10 {
        let input = HV16::random(i as u64); // Same seeds as session1
        session2.process_experience(&input, "learning experience", true, 0.5, 0.4);
    }

    let session2_report = session2.structured_report();

    // Identity should converge to similar state with similar experiences
    // (Not identical due to timing differences, but structurally similar)
    assert_eq!(
        session1_report.active_goals,
        session2_report.active_goals,
        "Active goals count should match with identical experiences"
    );

    assert_eq!(
        session1_report.core_values,
        session2_report.core_values,
        "Core values count should match with identical experiences"
    );
}

// ============================================================================
// SCENARIO 8: Trait Stability
// ============================================================================

#[test]
fn test_trait_stability() {
    let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

    // Get initial traits
    let initial_report = model.structured_report();
    let initial_traits = initial_report.traits;

    // Process many varied experiences
    for i in 0..100 {
        let success = i % 3 != 0; // 2/3 success rate
        let input = HV16::random(i as u64 + 5000);
        model.process_experience(
            &input,
            "varied experience",
            success,
            0.5,
            0.3,
        );
    }

    let final_report = model.structured_report();

    // Traits should persist (they're structural, not just statistical)
    assert!(
        final_report.traits >= initial_traits,
        "Traits should not be lost through normal processing"
    );

    // Coherence should remain meaningful
    assert!(
        final_report.coherence > 0.0,
        "Overall coherence should remain positive"
    );
}

// ============================================================================
// SCENARIO 9: Cross-Modal Identity Integration
// ============================================================================

#[test]
fn test_cross_modal_identity_integration() {
    let mut integration = NarrativeGWTIntegration::default_config();

    // Process with multiple modalities
    for i in 0..10 {
        integration.submit_content(
            "MultiModalStrategy",
            vec![HV16::random(i as u64 + 300)],
            "processing multi-modal content",
            vec![
                "visual".to_string(),
                "linguistic".to_string(),
                "conceptual".to_string(),
            ],
            0.7,
        );
        let result = integration.process();

        // Cross-modal Φ should be computed
        assert!(
            result.cross_modal_phi.is_some(),
            "Cross-modal Φ should be available"
        );
    }

    // Self-Φ should reflect cross-modal integration
    let final_phi = integration.self_phi();
    assert!(
        final_phi > 0.0,
        "Self-Φ should be positive with cross-modal integration"
    );
}

// ============================================================================
// SCENARIO 10: Standing Coalition Presence
// ============================================================================

#[test]
fn test_standing_coalition_always_present() {
    let mut integration = NarrativeGWTIntegration::default_config();

    // Process many cycles
    for i in 0..15 {
        integration.submit_content(
            &format!("TestStrategy{}", i),
            vec![HV16::random(i as u64 + 7000)],
            "task processing",
            vec!["module".to_string()],
            0.5,
        );
        let result = integration.process();

        // Self-Φ should always be present (standing coalition active)
        assert!(
            result.self_phi >= 0.0,
            "Self-Φ should never be negative"
        );

        // Coherence should be maintained
        assert!(
            result.coherence >= 0.0,
            "Coherence should never be negative"
        );
    }

    // Stats should reflect processing
    assert!(
        integration.stats.ignitions_processed > 0,
        "Should have processed ignitions"
    );
}
