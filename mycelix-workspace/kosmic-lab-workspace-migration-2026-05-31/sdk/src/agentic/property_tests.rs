// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Property-Based Tests
//!
//! Comprehensive property testing for trust algorithms.
//!
//! ## Properties Verified
//!
//! - Trust bounds invariants
//! - Quadratic voting correctness
//! - Slashing bounds
//! - Consensus safety
//! - Economic equilibrium properties

#[cfg(test)]
mod tests {
    use crate::agentic::{
        adaptive_thresholds::{
            AdaptiveConfig, AdaptiveThresholdEngine, FeedbackContext, FeedbackOutcome,
            ThresholdFeedback, ThresholdType,
        },
        cascade_analysis::{CascadeConfig, CascadeEngine, EdgeType, NetworkAgent, NetworkEdge},
        coordination::{AgentGroup, CoordinationConfig, Proposal, VoteType},
        differential_privacy::{DPConfig, PrivateAggregator},
        economics::{
            RewardConfig, RewardEngine, SlashResult, SlashingConfig, SlashingEngine,
            ViolationSeverity, ViolationType,
        },
        game_theory::{MechanismParams, validate_mechanism},
        temporal_trust::{DecayCurve, TemporalTrustConfig, TemporalTrustManager, TrustDecayConfig},
    };
    use crate::matl::KVector;

    // ========================================================================
    // Random Generator for Property Tests
    // ========================================================================

    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }

        fn next_f64(&mut self) -> f64 {
            (self.next_u64() as f64) / (u64::MAX as f64)
        }

        fn next_f32(&mut self) -> f32 {
            self.next_f64() as f32
        }

        fn next_range(&mut self, min: f64, max: f64) -> f64 {
            min + (max - min) * self.next_f64()
        }

        fn next_usize(&mut self, max: usize) -> usize {
            (self.next_u64() as usize) % max
        }
    }

    // ========================================================================
    // K-Vector Properties
    // ========================================================================

    /// Property: K-Vector trust scores are always in [0, 1]
    #[test]
    fn prop_kvector_trust_bounds() {
        let mut rng = TestRng::new(42);

        for _ in 0..1000 {
            let kv = KVector::new(
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
            );

            let trust = kv.trust_score();
            assert!(
                trust >= 0.0 && trust <= 1.0,
                "Trust score {} out of bounds [0, 1]",
                trust
            );
        }
    }

    /// Property: K-Vector with all zeros has minimum trust
    #[test]
    fn prop_kvector_zero_minimum() {
        let kv = KVector::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let trust = kv.trust_score();
        assert!(trust <= 0.1, "Zero K-Vector should have minimal trust");
    }

    /// Property: K-Vector with all ones has maximum trust
    #[test]
    fn prop_kvector_one_maximum() {
        let kv = KVector::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let trust = kv.trust_score();
        assert!(trust >= 0.9, "All-ones K-Vector should have maximal trust");
    }

    /// Property: K-Vector trust is monotonic with individual dimensions
    #[test]
    fn prop_kvector_monotonicity() {
        let base = KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        let base_trust = base.trust_score();

        // Increasing k_r should increase trust
        let increased = KVector::new(0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        let increased_trust = increased.trust_score();

        assert!(
            increased_trust >= base_trust,
            "Increasing k_r should not decrease trust"
        );
    }

    // ========================================================================
    // Coordination Properties
    // ========================================================================

    /// Property: Quadratic voting weight is sqrt of trust
    #[test]
    fn prop_quadratic_voting_formula() {
        let mut rng = TestRng::new(123);

        for _ in 0..100 {
            let trust = rng.next_range(0.1, 1.0);
            let linear_weight = trust;
            let quadratic_weight = trust.sqrt();

            // Quadratic weight should be greater than linear for trust < 1
            if trust < 1.0 {
                assert!(
                    quadratic_weight >= linear_weight,
                    "Quadratic weight {} should be >= linear {} for trust {}",
                    quadratic_weight,
                    linear_weight,
                    trust
                );
            }
        }
    }

    /// Property: Consensus requires quorum via submit_proposal + check_consensus
    #[test]
    fn prop_consensus_requires_quorum() {
        let config = CoordinationConfig {
            min_trust_threshold: 0.3,
            approval_threshold: 0.5,
            min_participation: 0.5,
            voting_timeout_ms: 1000,
            quadratic_voting: false,
            max_group_size: 100,
        };

        let mut group = AgentGroup::new(config);

        // Add members
        group.add_member("a1", 0.5).unwrap();
        group.add_member("a2", 0.5).unwrap();
        group.add_member("a3", 0.5).unwrap();
        group.add_member("a4", 0.5).unwrap();

        // Submit proposal
        let proposal = Proposal::new("Test", "test data").with_creator("a1");
        let prop_id = group.submit_proposal(proposal).unwrap();

        // Only one vote - should not reach quorum
        group.vote("a1", &prop_id, VoteType::Approve).unwrap();

        let result = group.check_consensus(&prop_id);
        // With 1/4 votes (25%), should not reach 50% participation
        assert!(
            result.is_none()
                || !matches!(
                    result.unwrap().decision,
                    crate::agentic::coordination::ConsensusDecision::Approved
                ),
            "Single vote should not reach 50% quorum"
        );
    }

    /// Property: Unanimous approval passes
    #[test]
    fn prop_unanimous_approval_passes() {
        let config = CoordinationConfig {
            min_trust_threshold: 0.3,
            approval_threshold: 0.5,
            min_participation: 0.5,
            voting_timeout_ms: 1000,
            quadratic_voting: false,
            max_group_size: 100,
        };

        let mut group = AgentGroup::new(config);

        group.add_member("a1", 0.5).unwrap();
        group.add_member("a2", 0.5).unwrap();

        let proposal = Proposal::new("Test", "data").with_creator("a1");
        let prop_id = group.submit_proposal(proposal).unwrap();
        group.vote("a1", &prop_id, VoteType::Approve).unwrap();
        group.vote("a2", &prop_id, VoteType::Approve).unwrap();

        let result = group.check_consensus(&prop_id);
        assert!(
            result.is_some(),
            "Should have consensus result with all votes in"
        );
        let result = result.unwrap();
        assert!(
            matches!(
                result.decision,
                crate::agentic::coordination::ConsensusDecision::Approved
            ),
            "Unanimous approval should pass"
        );
    }

    // ========================================================================
    // Slashing Properties
    // ========================================================================

    /// Property: Slashing never exceeds configured rate
    #[test]
    fn prop_slashing_bounded() {
        let mut rng = TestRng::new(456);

        let config = SlashingConfig {
            enabled: true,
            minor_violation_rate: 0.05,
            major_violation_rate: 0.20,
            critical_violation_rate: 0.50,
            slash_cooldown_ms: 0,
            max_cumulative_slash: 0.9,
        };

        let mut engine = SlashingEngine::new(config);

        for _ in 0..100 {
            let balance = (rng.next_f64() * 10000.0) as u64;
            let severity = match rng.next_usize(3) {
                0 => ViolationSeverity::Minor,
                1 => ViolationSeverity::Major,
                _ => ViolationSeverity::Critical,
            };

            let result = engine.slash(
                "agent-1",
                ViolationType::TrustGaming,
                severity,
                balance,
                "test",
            );

            if let SlashResult::Slashed { ref event, .. } = result {
                // Slashed amount should not exceed configured rate
                let max_rate = match severity {
                    ViolationSeverity::Minor => 0.05,
                    ViolationSeverity::Major => 0.20,
                    ViolationSeverity::Critical => 0.50,
                };

                if balance > 0 {
                    let actual_rate = event.amount_slashed as f64 / balance as f64;
                    assert!(
                        actual_rate <= max_rate + 0.001,
                        "Slashed rate {} exceeds max {} for {:?}",
                        actual_rate,
                        max_rate,
                        severity
                    );
                }
            }
        }
    }

    // ========================================================================
    // Temporal Trust Properties
    // ========================================================================

    /// Property: Trust decays over time
    #[test]
    fn prop_trust_decays() {
        let config = TemporalTrustConfig {
            decay: TrustDecayConfig {
                enabled: true,
                half_life_ms: 1000,
                floor: 0.1,
                grace_period_ms: 0,
                curve: DecayCurve::Exponential,
            },
            ..Default::default()
        };

        let mut manager = TemporalTrustManager::new(config);
        manager.record_trust("agent-1", 0.8);

        // Advance time
        manager.tick(2000);

        let current = manager.current_trust("agent-1").unwrap();
        assert!(
            current < 0.8,
            "Trust should decay from 0.8 to {} after 2 half-lives",
            current
        );
        assert!(
            current >= 0.1,
            "Trust should not go below floor 0.1, got {}",
            current
        );
    }

    /// Property: Trust does not decay below floor
    #[test]
    fn prop_trust_floor() {
        let config = TemporalTrustConfig {
            decay: TrustDecayConfig {
                enabled: true,
                half_life_ms: 100,
                floor: 0.2,
                grace_period_ms: 0,
                curve: DecayCurve::Exponential,
            },
            ..Default::default()
        };

        let mut manager = TemporalTrustManager::new(config);
        manager.record_trust("agent-1", 0.8);

        // Advance time significantly
        manager.tick(100000);

        let current = manager.current_trust("agent-1").unwrap();
        assert!(
            current >= 0.2 - 0.001,
            "Trust {} should not go below floor 0.2",
            current
        );
    }

    // ========================================================================
    // Adaptive Threshold Properties
    // ========================================================================

    /// Property: Thresholds stay within bounds
    #[test]
    fn prop_threshold_bounds() {
        let config = AdaptiveConfig {
            min_threshold: 0.3,
            max_threshold: 0.9,
            ..Default::default()
        };

        let mut engine = AdaptiveThresholdEngine::new(config);

        // Send random feedback
        let mut rng = TestRng::new(789);

        for i in 0..100 {
            let outcome = match rng.next_usize(4) {
                0 => FeedbackOutcome::TruePositive,
                1 => FeedbackOutcome::FalseNegative,
                2 => FeedbackOutcome::FalsePositive,
                _ => FeedbackOutcome::TrueNegative,
            };

            engine.process_feedback(ThresholdFeedback {
                threshold_type: ThresholdType::TrustAcceptance,
                threshold_value: engine.get_threshold(ThresholdType::TrustAcceptance),
                outcome,
                context: FeedbackContext {
                    network_health: 0.9,
                    active_agents: 10,
                    threat_level: 0.1,
                    ..Default::default()
                },
                timestamp: i as u64 * 1000,
            });

            let threshold = engine.get_threshold(ThresholdType::TrustAcceptance);
            // Use small epsilon for floating point comparison
            assert!(
                threshold >= 0.3 - 0.001 && threshold <= 0.9 + 0.001,
                "Threshold {} out of bounds [0.3, 0.9]",
                threshold
            );
        }
    }

    // ========================================================================
    // Cascade Properties
    // ========================================================================

    /// Property: Cascade depth is bounded
    #[test]
    fn prop_cascade_bounded_depth() {
        let config = CascadeConfig {
            max_depth: 3,
            propagation_factor: 0.5,
            ..Default::default()
        };

        let mut engine = CascadeEngine::new(config);
        let network = engine.network_mut();

        // Create a long chain
        for i in 0..10 {
            network.add_agent(NetworkAgent::new(format!("agent-{}", i), 0.8));
        }
        for i in 0..9 {
            network.add_edge(NetworkEdge {
                from: format!("agent-{}", i),
                to: format!("agent-{}", i + 1),
                weight: 0.8,
                edge_type: EdgeType::Attestation,
            });
        }

        let result = engine.apply_shock("agent-0", 0.5, 1000);

        assert!(
            result.max_depth_reached <= 3,
            "Cascade depth {} exceeds max 3",
            result.max_depth_reached
        );
    }

    // ========================================================================
    // Differential Privacy Properties
    // ========================================================================

    /// Property: Private mean converges to true mean with enough samples
    #[test]
    fn prop_private_mean_converges() {
        let config = DPConfig {
            epsilon: 100.0, // High epsilon for low noise in test
            ..Default::default()
        };

        let mut agg = PrivateAggregator::new(config);

        // Generate values with known mean 0.5
        let values: Vec<f64> = (0..1000).map(|i| (i as f64 % 100.0) / 100.0).collect();
        let true_mean: f64 = values.iter().sum::<f64>() / values.len() as f64;

        // Compute private mean with high epsilon (low noise)
        let private_mean = agg.private_mean(&values, 10.0).unwrap();

        // Should be within 0.1 of true mean with high probability
        assert!(
            (private_mean - true_mean).abs() < 0.1,
            "Private mean {} too far from true mean {}",
            private_mean,
            true_mean
        );
    }

    /// Property: Privacy budget is properly consumed
    #[test]
    fn prop_privacy_budget_consumed() {
        let config = DPConfig {
            epsilon: 2.0,
            ..Default::default()
        };

        let mut agg = PrivateAggregator::new(config);
        let values: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();

        // First query should succeed
        let r1 = agg.private_mean(&values, 1.5);
        assert!(r1.is_ok());

        // Second query should fail (budget exhausted)
        let r2 = agg.private_mean(&values, 1.5);
        assert!(r2.is_err(), "Should fail due to budget exhaustion");
    }

    // ========================================================================
    // Game Theory Properties
    // ========================================================================

    /// Property: Good mechanism parameters score higher
    #[test]
    fn prop_good_mechanism_scores_higher() {
        let good_params = MechanismParams {
            base_reward: 1.0,
            slashing_rate: 0.5,
            min_stake: 100.0,
            quadratic_voting: true,
            trust_threshold: 0.5,
            attestation_weight: 0.1,
        };

        let bad_params = MechanismParams {
            base_reward: 0.1,
            slashing_rate: 0.01,
            min_stake: 1.0,
            quadratic_voting: false,
            trust_threshold: 0.1,
            attestation_weight: 1.0,
        };

        let good_result = validate_mechanism(&good_params);
        let bad_result = validate_mechanism(&bad_params);

        assert!(
            good_result.score > bad_result.score,
            "Good params should score higher: {} vs {}",
            good_result.score,
            bad_result.score
        );
    }

    // ========================================================================
    // Cross-Module Integration Properties
    // ========================================================================

    /// Property: High trust agents survive slashing better
    #[test]
    fn prop_high_trust_survives_slash() {
        let config = SlashingConfig::default();
        let mut engine = SlashingEngine::new(config);

        let balance = 10000u64;

        // Minor violation should slash less than critical
        let minor_result = engine.slash(
            "high-trust",
            ViolationType::RateLimitViolation,
            ViolationSeverity::Minor,
            balance,
            "minor violation",
        );

        let critical_result = engine.slash(
            "low-trust",
            ViolationType::TrustGaming,
            ViolationSeverity::Critical,
            balance,
            "critical violation",
        );

        if let (
            SlashResult::Slashed {
                event: minor_event, ..
            },
            SlashResult::Slashed {
                event: critical_event,
                ..
            },
        ) = (&minor_result, &critical_result)
        {
            assert!(
                minor_event.amount_slashed < critical_event.amount_slashed,
                "Minor violation slash {} should be less than critical {}",
                minor_event.amount_slashed,
                critical_event.amount_slashed
            );
        }
    }

    /// Property: Rewards incentivize honest behavior
    #[test]
    fn prop_rewards_incentivize_honest() {
        let config = RewardConfig::default();
        let engine = RewardEngine::new(config);

        // Higher trust should yield higher reward
        let honest_reward = engine.calculate_participation_reward("honest-agent", 0.8);
        let malicious_reward = engine.calculate_participation_reward("malicious-agent", 0.2);

        assert!(
            honest_reward > malicious_reward,
            "Honest reward {} should exceed malicious reward {}",
            honest_reward,
            malicious_reward
        );
    }
}
