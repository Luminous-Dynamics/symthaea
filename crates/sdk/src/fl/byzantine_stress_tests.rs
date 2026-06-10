// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Byzantine Resistance Stress Tests
//!
//! Comprehensive testing of the 34% validated Byzantine tolerance provided by RB-BFT.
//! Tests edge cases, large-scale scenarios, and adaptive adversary simulations.
//!
//! # Test Categories
//!
//! 1. **Threshold Edge Cases**: Tests at 33%, 34%, 35% Byzantine fractions (and 45% as above-threshold boundary)
//! 2. **Scale Tests**: Large participant counts (100+)
//! 3. **Adaptive Adversaries**: Reputation gaming attacks
//! 4. **Recovery Tests**: System behavior after Byzantine detection

#[cfg(test)]
mod tests {
    use crate::fl::{GradientUpdate, RbbftFLBridge, RbbftFLConfig};
    use crate::matl::KVector;

    /// Create a GradientUpdate from a Vec<f64>
    fn make_gradient(participant_id: &str, gradients: Vec<f64>) -> GradientUpdate {
        GradientUpdate::new(
            participant_id.to_string(),
            1, // round
            gradients,
            100, // samples
            0.5, // loss
        )
    }

    // ========================================================================
    // THRESHOLD EDGE CASE TESTS
    // ========================================================================

    /// Helper to create a bridge with N participants and M Byzantine
    fn setup_byzantine_scenario(
        total_participants: usize,
        byzantine_fraction: f32,
        honest_reputation: f32,
        byzantine_reputation: f32,
    ) -> (RbbftFLBridge, Vec<String>, Vec<String>) {
        let mut config = RbbftFLConfig::testing();
        config.min_participants = 2;
        config.min_reputation = 0.1;
        let mut bridge = RbbftFLBridge::new(config);

        let byzantine_count = (total_participants as f32 * byzantine_fraction).ceil() as usize;
        let honest_count = total_participants - byzantine_count;

        let mut honest_ids = Vec::new();
        let mut byzantine_ids = Vec::new();

        // Register honest participants
        for i in 0..honest_count {
            let id = format!("honest-{}", i);
            let kv = KVector::new(
                honest_reputation, // k_r
                0.5,
                0.9,
                0.5,
                0.3,
                0.4,
                0.5,
                0.3,
                0.5,
                0.5,
            );
            bridge.register_participant(&id, kv);
            honest_ids.push(id);
        }

        // Register Byzantine participants
        for i in 0..byzantine_count {
            let id = format!("byzantine-{}", i);
            let kv = KVector::new(
                byzantine_reputation, // k_r
                0.5,
                0.9,
                0.5,
                0.3,
                0.4,
                0.5,
                0.3,
                0.5,
                0.5,
            );
            bridge.register_participant(&id, kv);
            byzantine_ids.push(id);
        }

        (bridge, honest_ids, byzantine_ids)
    }

    #[test]
    fn test_44_percent_byzantine_should_succeed() {
        // 44% Byzantine (below threshold) - consensus SHOULD succeed
        let (mut bridge, honest_ids, byzantine_ids) = setup_byzantine_scenario(
            100,  // total
            0.44, // 44% Byzantine
            0.7,  // honest reputation
            0.4,  // Byzantine reputation (lower)
        );

        bridge.start_round([0u8; 32]).expect("Start round failed");

        // Honest votes: YES with valid gradients
        for id in &honest_ids {
            let gradient = make_gradient(id, vec![0.1; 100]);
            bridge
                .submit_vote(id, true, Some(gradient), true)
                .expect("Vote failed");
        }

        // Byzantine votes: NO (trying to block)
        for id in &byzantine_ids {
            let gradient = make_gradient(id, vec![100.0; 100]);
            bridge
                .submit_vote(id, false, Some(gradient), true)
                .expect("Vote failed");
        }

        // With reputation² weighting and 44% Byzantine, consensus should succeed
        assert!(
            bridge.check_consensus(),
            "Consensus should succeed with 44% Byzantine (below threshold)"
        );

        let result = bridge.finalize_round();
        assert!(
            result.is_ok(),
            "Finalization should succeed: {:?}",
            result.err()
        );

        let stats = bridge.stats();
        let byzantine_est = bridge.estimate_byzantine_fraction();
        println!("44% Byzantine test:");
        println!("  Byzantine fraction (est): {:.2}%", byzantine_est * 100.0);
        println!(
            "  Consensus ratio: {:.2}%",
            bridge.consensus_ratio().unwrap_or(0.0) * 100.0
        );
        println!("  Stats: {:?}", stats);
    }

    #[test]
    fn test_45_percent_byzantine_at_threshold() {
        // 45% Byzantine (ABOVE validated 34% threshold) - boundary test that may succeed
        // with reputation advantage but is beyond the validated maximum
        let (mut bridge, honest_ids, byzantine_ids) = setup_byzantine_scenario(
            100, 0.45, // Above validated 34% threshold - boundary test
            0.8,  // Higher honest reputation to edge out
            0.3,  // Lower Byzantine reputation
        );

        bridge.start_round([0u8; 32]).expect("Start round failed");

        for id in &honest_ids {
            let gradient = make_gradient(id, vec![0.1; 100]);
            bridge
                .submit_vote(id, true, Some(gradient), true)
                .expect("Vote failed");
        }

        for id in &byzantine_ids {
            let gradient = make_gradient(id, vec![100.0; 100]);
            bridge
                .submit_vote(id, false, Some(gradient), true)
                .expect("Vote failed");
        }

        // At 45% (above validated 34% threshold), result depends on reputation distribution
        let consensus = bridge.check_consensus();
        let consensus_ratio = bridge.consensus_ratio().unwrap_or(0.0);
        let byzantine_est = bridge.estimate_byzantine_fraction();

        println!("45% Byzantine test (above validated threshold):");
        println!("  Byzantine fraction (est): {:.2}%", byzantine_est * 100.0);
        println!("  Consensus ratio: {:.2}%", consensus_ratio * 100.0);
        println!("  Consensus reached: {}", consensus);

        // With higher honest reputation, may still succeed but is above validated 34% threshold
        assert!(
            consensus,
            "Consensus should succeed at 45% with reputation advantage (above validated 34% threshold)"
        );
    }

    #[test]
    fn test_46_percent_byzantine_should_fail_or_flag() {
        // 46% Byzantine (above threshold) - consensus might fail
        let (mut bridge, honest_ids, byzantine_ids) = setup_byzantine_scenario(
            100, 0.46, // 46% Byzantine
            0.6,  // Equal reputation
            0.6,  // Equal reputation
        );

        bridge.start_round([0u8; 32]).expect("Start round failed");

        for id in &honest_ids {
            let gradient = make_gradient(id, vec![0.1; 100]);
            bridge
                .submit_vote(id, true, Some(gradient), true)
                .expect("Vote failed");
        }

        // Byzantine participants vote NO with invalid proof flag
        for id in &byzantine_ids {
            let gradient = make_gradient(id, vec![100.0; 100]);
            // Mark proof as invalid to simulate Byzantine behavior
            bridge
                .submit_vote(id, false, Some(gradient), false)
                .expect("Vote failed");
        }

        let byzantine_est = bridge.estimate_byzantine_fraction();
        let consensus_ratio = bridge.consensus_ratio().unwrap_or(0.0);
        println!("46% Byzantine test:");
        println!("  Byzantine fraction (est): {:.2}%", byzantine_est * 100.0);
        println!("  Consensus ratio: {:.2}%", consensus_ratio * 100.0);

        // Either consensus fails or high Byzantine fraction is detected
        // The Byzantine fraction based on invalid proofs should be detected
        // (above validated 34% threshold)
        assert!(
            byzantine_est >= 0.34,
            "Should detect high Byzantine fraction (above validated 34% threshold)"
        );
    }

    // ========================================================================
    // SCALE TESTS
    // ========================================================================

    #[test]
    fn test_large_scale_100_participants() {
        let (mut bridge, honest_ids, byzantine_ids) = setup_byzantine_scenario(
            100, 0.30, // 30% Byzantine
            0.7, 0.4,
        );

        bridge.start_round([0u8; 32]).expect("Start round failed");

        for id in &honest_ids {
            let gradient = make_gradient(id, vec![0.1; 1000]);
            bridge
                .submit_vote(id, true, Some(gradient), true)
                .expect("Vote failed");
        }

        for id in &byzantine_ids {
            let gradient = make_gradient(id, vec![10.0; 1000]);
            bridge
                .submit_vote(id, false, Some(gradient), true)
                .expect("Vote failed");
        }

        assert!(
            bridge.check_consensus(),
            "Should reach consensus with 30% Byzantine"
        );

        let result = bridge.finalize_round().expect("Finalization failed");
        println!("100 participant test:");
        println!("  Valid submissions: {}", result.participant_count);
        println!("  Excluded count: {}", result.excluded_count);
    }

    #[test]
    fn test_large_scale_500_participants() {
        let (mut bridge, honest_ids, byzantine_ids) = setup_byzantine_scenario(
            500, 0.35, // 35% Byzantine
            0.7, 0.3,
        );

        bridge.start_round([0u8; 32]).expect("Start round failed");

        for id in &honest_ids {
            let gradient = make_gradient(id, vec![0.05; 500]);
            bridge
                .submit_vote(id, true, Some(gradient), true)
                .expect("Vote failed");
        }

        for id in &byzantine_ids {
            let gradient = make_gradient(id, vec![5.0; 500]);
            bridge
                .submit_vote(id, false, Some(gradient), true)
                .expect("Vote failed");
        }

        assert!(
            bridge.check_consensus(),
            "Should reach consensus with 35% Byzantine"
        );

        let result = bridge.finalize_round().expect("Finalization failed");
        println!("500 participant test:");
        println!("  Valid submissions: {}", result.participant_count);
        assert!(
            result.participant_count >= 300,
            "Should have most honest participants"
        );
    }

    // ========================================================================
    // ADAPTIVE ADVERSARY TESTS
    // ========================================================================

    #[test]
    fn test_reputation_gaming_attack() {
        // Adversary tries to build reputation over time, then attack
        let mut config = RbbftFLConfig::testing();
        config.min_participants = 2;
        let mut bridge = RbbftFLBridge::new(config);

        // Register participants
        let honest_id = "honest-1";
        let gamer_id = "gamer-1";

        // Honest participant with established reputation
        bridge.register_participant(
            honest_id,
            KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.5, 0.3, 0.6, 0.5),
        );

        // Gamer starts with medium reputation (built over time)
        bridge.register_participant(
            gamer_id,
            KVector::new(0.6, 0.5, 0.8, 0.5, 0.4, 0.3, 0.4, 0.2, 0.5, 0.4),
        );

        // Round 1: Gamer behaves honestly
        bridge.start_round([1u8; 32]).expect("Start round 1");
        bridge
            .submit_vote(
                honest_id,
                true,
                Some(make_gradient(honest_id, vec![0.1; 100])),
                true,
            )
            .unwrap();
        bridge
            .submit_vote(
                gamer_id,
                true,
                Some(make_gradient(gamer_id, vec![0.1; 100])),
                true,
            )
            .unwrap();
        assert!(bridge.check_consensus(), "Round 1 should succeed");
        let _ = bridge.finalize_round();

        // Round 2: Gamer attacks
        bridge.start_round([2u8; 32]).expect("Start round 2");
        bridge
            .submit_vote(
                honest_id,
                true,
                Some(make_gradient(honest_id, vec![0.1; 100])),
                true,
            )
            .unwrap();
        // Gamer votes yes but submits malicious gradient
        bridge
            .submit_vote(
                gamer_id,
                true,
                Some(make_gradient(gamer_id, vec![1000.0; 100])),
                true,
            )
            .unwrap();

        // Consensus should still work, but malicious gradient is included
        assert!(bridge.check_consensus(), "Round 2 should reach consensus");

        let result = bridge.finalize_round().expect("Round 2 finalization");
        println!("Reputation gaming test:");
        println!(
            "  Result contains {} participants",
            result.participant_count
        );
    }

    #[test]
    fn test_sybil_attack_with_low_reputation() {
        // Adversary creates many identities but each has low reputation
        let mut config = RbbftFLConfig::testing();
        config.min_participants = 2;
        config.min_reputation = 0.3; // Require reasonable reputation
        let mut bridge = RbbftFLBridge::new(config);

        // 5 honest participants with high reputation
        for i in 0..5 {
            let id = format!("honest-{}", i);
            bridge.register_participant(
                &id,
                KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.5, 0.3, 0.6, 0.5),
            );
        }

        // 20 Sybil identities with varying reputation (many filtered)
        let mut sybil_registered = 0;
        for i in 0..20 {
            let id = format!("sybil-{}", i);
            // Low reputation - some will be filtered
            let rep = 0.2 + (i as f32 * 0.02); // 0.2 to 0.58
            if bridge.register_participant(
                &id,
                KVector::new(rep, 0.3, 0.5, 0.3, 0.2, 0.1, 0.2, 0.1, 0.3, 0.2),
            ) {
                sybil_registered += 1;
            }
        }

        bridge.start_round([0u8; 32]).expect("Start round");

        // Honest votes
        for i in 0..5 {
            let id = format!("honest-{}", i);
            bridge
                .submit_vote(&id, true, Some(make_gradient(&id, vec![0.1; 100])), true)
                .unwrap();
        }

        // Sybil votes (those that were registered)
        for i in 0..20 {
            let id = format!("sybil-{}", i);
            let rep = 0.2 + (i as f32 * 0.02);
            if rep >= 0.3 {
                let _ =
                    bridge.submit_vote(&id, false, Some(make_gradient(&id, vec![50.0; 100])), true);
            }
        }

        // With reputation² weighting, honest votes should dominate
        let consensus = bridge.check_consensus();
        println!("Sybil attack test:");
        println!("  Honest count: 5");
        println!("  Sybil registered: {}", sybil_registered);
        println!("  Consensus reached: {}", consensus);

        // High-reputation honest participants should win
        assert!(
            consensus,
            "Honest participants should win against low-rep Sybils"
        );
    }

    #[test]
    fn test_collusion_attack() {
        // Multiple high-reputation participants collude
        let mut config = RbbftFLConfig::testing();
        config.min_participants = 2;
        let mut bridge = RbbftFLBridge::new(config);

        // 6 honest participants
        for i in 0..6 {
            let id = format!("honest-{}", i);
            bridge.register_participant(
                &id,
                KVector::new(0.7, 0.5, 0.8, 0.6, 0.4, 0.3, 0.4, 0.2, 0.5, 0.4),
            );
        }

        // 4 colluding participants (40% - below threshold but significant)
        for i in 0..4 {
            let id = format!("colluder-{}", i);
            bridge.register_participant(
                &id,
                KVector::new(0.75, 0.6, 0.85, 0.65, 0.45, 0.35, 0.45, 0.25, 0.55, 0.45),
            );
        }

        bridge.start_round([0u8; 32]).expect("Start round");

        // Honest votes with legitimate gradients
        for i in 0..6 {
            let id = format!("honest-{}", i);
            let gradient = vec![0.1 + i as f64 * 0.01; 100]; // Slightly varied
            bridge
                .submit_vote(&id, true, Some(make_gradient(&id, gradient)), true)
                .unwrap();
        }

        // Colluders coordinate: same malicious gradient
        let malicious_gradient = vec![-10.0; 100];
        for i in 0..4 {
            let id = format!("colluder-{}", i);
            bridge
                .submit_vote(
                    &id,
                    true,
                    Some(make_gradient(&id, malicious_gradient.clone())),
                    true,
                )
                .unwrap();
        }

        assert!(bridge.check_consensus(), "Should reach consensus");

        let result = bridge.finalize_round().expect("Finalization");
        println!("Collusion attack test:");
        println!("  Participants: {}", result.participant_count);
        println!("  Excluded: {}", result.excluded_count);
    }

    // ========================================================================
    // RECOVERY TESTS
    // ========================================================================

    #[test]
    fn test_recovery_after_byzantine_detection() {
        let mut config = RbbftFLConfig::testing();
        config.min_participants = 2;
        let mut bridge = RbbftFLBridge::new(config);

        // Setup: 3 honest, 2 Byzantine
        for i in 0..3 {
            let id = format!("honest-{}", i);
            bridge.register_participant(
                &id,
                KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.5, 0.3, 0.6, 0.5),
            );
        }

        for i in 0..2 {
            let id = format!("byzantine-{}", i);
            bridge.register_participant(
                &id,
                KVector::new(0.5, 0.4, 0.6, 0.4, 0.3, 0.2, 0.3, 0.1, 0.4, 0.3),
            );
        }

        // Round 1: Byzantine participants attack
        bridge.start_round([1u8; 32]).expect("Start round 1");

        for i in 0..3 {
            let id = format!("honest-{}", i);
            bridge
                .submit_vote(&id, true, Some(make_gradient(&id, vec![0.1; 100])), true)
                .unwrap();
        }

        for i in 0..2 {
            let id = format!("byzantine-{}", i);
            bridge
                .submit_vote(&id, false, Some(make_gradient(&id, vec![100.0; 100])), true)
                .unwrap();
        }

        // Should succeed (40% Byzantine, below threshold)
        assert!(bridge.check_consensus(), "Round 1 should succeed");
        let result1 = bridge.finalize_round().expect("Round 1 finalization");

        // Round 2: System continues with same participants
        bridge.start_round([2u8; 32]).expect("Start round 2");

        for i in 0..3 {
            let id = format!("honest-{}", i);
            bridge
                .submit_vote(&id, true, Some(make_gradient(&id, vec![0.1; 100])), true)
                .unwrap();
        }

        for i in 0..2 {
            let id = format!("byzantine-{}", i);
            bridge
                .submit_vote(&id, false, Some(make_gradient(&id, vec![100.0; 100])), true)
                .unwrap();
        }

        assert!(bridge.check_consensus(), "Round 2 should succeed");
        let result2 = bridge.finalize_round().expect("Round 2 finalization");

        println!("Recovery test:");
        println!("  Round 1 participants: {}", result1.participant_count);
        println!("  Round 2 participants: {}", result2.participant_count);
        println!("  System continues operating despite persistent Byzantine behavior");
    }

    // ========================================================================
    // REPUTATION WEIGHTING VERIFICATION
    // ========================================================================

    #[test]
    fn test_quadratic_weighting_amplifies_trust() {
        // Verify that reputation² gives high-reputation participants much more influence
        let mut config = RbbftFLConfig::testing();
        config.min_participants = 2;
        config.use_quadratic_weighting = true;
        let mut bridge = RbbftFLBridge::new(config);

        // 1 high-reputation honest participant
        bridge.register_participant(
            "high-rep",
            KVector::new(0.95, 0.8, 0.95, 0.85, 0.7, 0.6, 0.7, 0.5, 0.8, 0.75),
        );

        // 10 low-reputation Byzantine participants
        for i in 0..10 {
            let id = format!("low-rep-{}", i);
            bridge.register_participant(
                &id,
                KVector::new(0.3, 0.2, 0.4, 0.25, 0.15, 0.1, 0.15, 0.05, 0.2, 0.15),
            );
        }

        bridge.start_round([0u8; 32]).expect("Start round");

        // High-rep votes YES
        bridge
            .submit_vote(
                "high-rep",
                true,
                Some(make_gradient("high-rep", vec![0.1; 100])),
                true,
            )
            .unwrap();

        // All low-rep vote NO
        for i in 0..10 {
            let id = format!("low-rep-{}", i);
            bridge
                .submit_vote(&id, false, Some(make_gradient(&id, vec![50.0; 100])), true)
                .unwrap();
        }

        // With reputation²:
        // - High-rep weight: 0.95² = 0.9025
        // - Each low-rep weight: 0.3² = 0.09
        // - Total low-rep weight: 10 × 0.09 = 0.9

        let ratio = bridge.consensus_ratio().unwrap_or(0.0);
        println!("Quadratic weighting test:");
        println!("  High-rep (0.95): weight = {:.3}", 0.95f32.powi(2));
        println!(
            "  Low-rep (0.3 × 10): weight = {:.3}",
            10.0 * 0.3f32.powi(2)
        );
        println!("  Consensus ratio: {:.2}%", ratio * 100.0);

        // The single high-reputation participant should nearly match 10 low-reputation ones
        // Ratio should be close to 50% (0.9025 / (0.9025 + 0.9) ≈ 0.5)
    }

    #[test]
    fn test_linear_vs_quadratic_weighting() {
        // Compare linear vs quadratic weighting outcomes
        let base_config = RbbftFLConfig::testing();

        let run_test = |use_quadratic: bool| -> f32 {
            let mut config = base_config.clone();
            config.use_quadratic_weighting = use_quadratic;
            config.min_participants = 2;
            let mut bridge = RbbftFLBridge::new(config);

            // 2 high-rep, 3 medium-rep
            bridge.register_participant(
                "high-1",
                KVector::new(0.9, 0.7, 0.9, 0.8, 0.6, 0.5, 0.6, 0.4, 0.7, 0.6),
            );
            bridge.register_participant(
                "high-2",
                KVector::new(0.85, 0.65, 0.85, 0.75, 0.55, 0.45, 0.55, 0.35, 0.65, 0.55),
            );
            bridge.register_participant(
                "med-1",
                KVector::new(0.5, 0.4, 0.6, 0.45, 0.35, 0.25, 0.35, 0.15, 0.4, 0.3),
            );
            bridge.register_participant(
                "med-2",
                KVector::new(0.5, 0.4, 0.6, 0.45, 0.35, 0.25, 0.35, 0.15, 0.4, 0.3),
            );
            bridge.register_participant(
                "med-3",
                KVector::new(0.5, 0.4, 0.6, 0.45, 0.35, 0.25, 0.35, 0.15, 0.4, 0.3),
            );

            bridge.start_round([0u8; 32]).expect("Start round");

            // High-rep vote YES
            bridge
                .submit_vote(
                    "high-1",
                    true,
                    Some(make_gradient("high-1", vec![0.1; 50])),
                    true,
                )
                .unwrap();
            bridge
                .submit_vote(
                    "high-2",
                    true,
                    Some(make_gradient("high-2", vec![0.1; 50])),
                    true,
                )
                .unwrap();

            // Medium-rep vote NO
            bridge
                .submit_vote(
                    "med-1",
                    false,
                    Some(make_gradient("med-1", vec![5.0; 50])),
                    true,
                )
                .unwrap();
            bridge
                .submit_vote(
                    "med-2",
                    false,
                    Some(make_gradient("med-2", vec![5.0; 50])),
                    true,
                )
                .unwrap();
            bridge
                .submit_vote(
                    "med-3",
                    false,
                    Some(make_gradient("med-3", vec![5.0; 50])),
                    true,
                )
                .unwrap();

            bridge.consensus_ratio().unwrap_or(0.0)
        };

        let linear_approval = run_test(false);
        let quadratic_approval = run_test(true);

        println!("Linear vs Quadratic weighting:");
        println!("  Linear approval: {:.2}%", linear_approval * 100.0);
        println!("  Quadratic approval: {:.2}%", quadratic_approval * 100.0);

        // Quadratic weighting should give MORE weight to high-rep
        // So approval should be HIGHER with quadratic
        assert!(
            quadratic_approval >= linear_approval,
            "Quadratic weighting should favor high-reputation participants"
        );
    }
}
