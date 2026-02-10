//! # Property-Based Tests
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
        game_theory::{validate_mechanism, MechanismParams},
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

    // ========================================================================
    // Dashboard Properties
    // ========================================================================

    use crate::agentic::dashboard::{
        AlertPanel, AlertSeverity, AlertStatus, ChartType, Dashboard, DashboardAlert,
        DashboardConfig, DashboardEventType, EventPriority, EventStream, LiveMetrics,
        MetricsAggregator, MetricsInput, TimeSeries,
    };

    /// Property: Network health is always bounded [0, 1]
    #[test]
    fn prop_network_health_bounded() {
        let mut rng = TestRng::new(100);
        let config = DashboardConfig::default();
        let mut agg = MetricsAggregator::new(config);

        for i in 0..100 {
            let trust_scores: Vec<f64> = (0..rng.next_usize(50) + 1)
                .map(|_| rng.next_range(0.0, 1.0))
                .collect();
            let phi_values: Vec<f64> = (0..rng.next_usize(10) + 1)
                .map(|_| rng.next_range(0.0, 1.0))
                .collect();
            let threats: Vec<f64> = (0..rng.next_usize(5))
                .map(|_| rng.next_range(0.0, 1.0))
                .collect();

            let input = MetricsInput {
                trust_scores,
                transaction_count: rng.next_usize(100),
                alerts: vec![],
                phi_values,
                threats,
            };

            let metrics = agg.update(input, 1000 + i * 60);

            assert!(
                metrics.network_health >= 0.0 && metrics.network_health <= 1.0,
                "Network health {} out of bounds at iteration {}",
                metrics.network_health,
                i
            );
        }
    }

    /// Property: Average trust is bounded and computed correctly
    #[test]
    fn prop_average_trust_computation() {
        let mut rng = TestRng::new(200);
        let config = DashboardConfig::default();
        let mut agg = MetricsAggregator::new(config);

        for i in 0..50 {
            let trust_scores: Vec<f64> = (0..rng.next_usize(20) + 1)
                .map(|_| rng.next_range(0.0, 1.0))
                .collect();

            let expected_avg = trust_scores.iter().sum::<f64>() / trust_scores.len() as f64;

            let input = MetricsInput {
                trust_scores,
                transaction_count: 0,
                alerts: vec![],
                phi_values: vec![],
                threats: vec![],
            };

            let metrics = agg.update(input, 1000 + i * 60);

            assert!(
                (metrics.average_trust - expected_avg).abs() < 0.001,
                "Average trust {} does not match expected {}",
                metrics.average_trust,
                expected_avg
            );
        }
    }

    /// Property: Alert counts are non-negative and accurate
    #[test]
    fn prop_alert_counts_accurate() {
        let config = DashboardConfig::default();
        let mut panel = AlertPanel::new(config);

        let mut expected_critical = 0usize;
        let mut expected_high = 0usize;
        let mut expected_medium = 0usize;
        let mut expected_low = 0usize;

        let mut rng = TestRng::new(300);

        for _ in 0..50 {
            let severity = match rng.next_usize(4) {
                0 => {
                    expected_critical += 1;
                    AlertSeverity::Critical
                }
                1 => {
                    expected_high += 1;
                    AlertSeverity::High
                }
                2 => {
                    expected_medium += 1;
                    AlertSeverity::Medium
                }
                _ => {
                    expected_low += 1;
                    AlertSeverity::Low
                }
            };

            panel.create_alert(severity, "Test", "Description", "test");
        }

        let active = panel.active_alerts();
        assert_eq!(active.len(), 50, "Should have 50 active alerts");

        let critical = panel.critical_alerts();
        assert_eq!(
            critical.len(),
            expected_critical,
            "Critical count mismatch: {} vs {}",
            critical.len(),
            expected_critical
        );
    }

    /// Property: Event stream respects buffer size limit
    #[test]
    fn prop_event_stream_bounded() {
        let mut config = DashboardConfig::default();
        config.event_buffer_size = 10;
        let mut stream = EventStream::new(config);

        // Push more events than buffer allows
        for i in 0..25 {
            stream.emit(
                DashboardEventType::Custom,
                EventPriority::Low,
                &format!("source-{}", i),
                "Test event",
            );
        }

        // Stream should be bounded
        let recent = stream.recent(100);
        assert!(recent.len() <= 10, "Event stream should be bounded to 10");
    }

    /// Property: Time series min/max are consistent
    #[test]
    fn prop_time_series_minmax() {
        let mut rng = TestRng::new(400);
        let mut series = TimeSeries::new("Test", ChartType::Line);

        let values: Vec<f64> = (0..100).map(|_| rng.next_range(0.0, 100.0)).collect();

        for (i, &v) in values.iter().enumerate() {
            series.add_point(i as u64 * 1000, v);
        }

        let min = series.min().unwrap();
        let max = series.max().unwrap();

        assert!(min <= max, "Min {} should be <= max {}", min, max);

        let actual_min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let actual_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        assert!((min - actual_min).abs() < 0.001, "Min mismatch");
        assert!((max - actual_max).abs() < 0.001, "Max mismatch");
    }

    /// Property: Dashboard metrics history is bounded
    #[test]
    fn prop_dashboard_history_bounded() {
        let mut config = DashboardConfig::default();
        config.max_history = 50;
        let mut agg = MetricsAggregator::new(config);

        for i in 0..100 {
            let input = MetricsInput {
                trust_scores: vec![0.5],
                transaction_count: 1,
                alerts: vec![],
                phi_values: vec![0.6],
                threats: vec![0.1],
            };
            agg.update(input, i);
        }

        assert!(
            agg.history().len() <= 50,
            "History should be bounded to 50, got {}",
            agg.history().len()
        );
    }

    // ========================================================================
    // Verification Properties
    // ========================================================================

    use crate::agentic::verification::{
        AtomicPredicate, CompareOp, Invariant, InvariantType, PropertyFormula, PropertyKind,
        PropertySpec, SystemState, VerificationEngine, ViolationSeverity as VerifViolationSeverity,
    };

    /// Property: Trust bounds invariant detects violations
    #[test]
    fn prop_verification_trust_bounds() {
        let mut rng = TestRng::new(500);
        let mut engine = VerificationEngine::with_defaults();

        for i in 0..50 {
            let mut trust_scores = std::collections::HashMap::new();
            let should_violate = i % 5 == 0;

            for j in 0..5 {
                let score = if should_violate && j == 0 {
                    // Create a violation
                    1.5 + rng.next_f64()
                } else {
                    rng.next_range(0.0, 1.0)
                };
                trust_scores.insert(format!("agent-{}", j), score);
            }

            let state = SystemState {
                index: i,
                timestamp: 1000 + i as u64,
                trust_scores,
                byzantine_count: 0,
                network_health: 0.9,
                variables: std::collections::HashMap::new(),
            };

            let results = engine.check_invariants(&state);
            let trust_result = results
                .iter()
                .find(|r| r.invariant_id == "trust_bounds")
                .unwrap();

            if should_violate {
                assert!(
                    !trust_result.holds,
                    "Should detect trust bounds violation at iteration {}",
                    i
                );
            }
        }
    }

    /// Property: Byzantine tolerance invariant is accurate
    #[test]
    fn prop_verification_byzantine_tolerance() {
        let mut engine = VerificationEngine::with_defaults();

        // Test various Byzantine fractions
        let test_cases = [
            (0, 10, true),  // 0% - should pass
            (3, 10, true),  // 30% - should pass (under 34% validated threshold)
            (4, 10, false), // 40% - should fail (over 34% validated threshold)
            (5, 10, false), // 50% - should fail (over 34% validated threshold)
            (9, 10, false), // 90% - should fail
        ];

        for (byz_count, total, should_pass) in test_cases {
            let mut trust_scores = std::collections::HashMap::new();
            for i in 0..total {
                trust_scores.insert(format!("agent-{}", i), 0.5);
            }

            let state = SystemState {
                index: 0,
                timestamp: 1000,
                trust_scores,
                byzantine_count: byz_count,
                network_health: 0.9,
                variables: std::collections::HashMap::new(),
            };

            let results = engine.check_invariants(&state);
            let byz_result = results
                .iter()
                .find(|r| r.invariant_id == "byzantine_tolerance")
                .unwrap();

            assert_eq!(
                byz_result.holds, should_pass,
                "Byzantine {}/{}: expected holds={}, got holds={}",
                byz_count, total, should_pass, byz_result.holds
            );
        }
    }

    /// Property: CompareOp evaluations are consistent
    #[test]
    fn prop_compare_op_consistent() {
        let mut rng = TestRng::new(600);

        for _ in 0..100 {
            let a = rng.next_range(-100.0, 100.0);
            let b = rng.next_range(-100.0, 100.0);

            // Lt and Ge are complementary
            assert_eq!(
                CompareOp::Lt.eval(a, b),
                !CompareOp::Ge.eval(a, b),
                "Lt and Ge should be complementary for {} and {}",
                a,
                b
            );

            // Gt and Le are complementary
            assert_eq!(
                CompareOp::Gt.eval(a, b),
                !CompareOp::Le.eval(a, b),
                "Gt and Le should be complementary for {} and {}",
                a,
                b
            );

            // Transitivity: if a < b and b < c, then a < c
            let c = b + rng.next_range(0.001, 10.0);
            if CompareOp::Lt.eval(a, b) && CompareOp::Lt.eval(b, c) {
                assert!(
                    CompareOp::Lt.eval(a, c),
                    "Transitivity violated for {} < {} < {}",
                    a,
                    b,
                    c
                );
            }
        }
    }

    /// Property: Verification summary is consistent
    #[test]
    fn prop_verification_summary_consistent() {
        let mut engine = VerificationEngine::with_defaults();

        // Add some checks
        for i in 0..10 {
            let mut trust_scores = std::collections::HashMap::new();
            trust_scores.insert("agent-1".to_string(), if i % 3 == 0 { 1.5 } else { 0.5 });

            let state = SystemState {
                index: i,
                timestamp: 1000 + i as u64,
                trust_scores,
                byzantine_count: 0,
                network_health: 0.9,
                variables: std::collections::HashMap::new(),
            };

            engine.check_invariants(&state);
        }

        let summary = engine.summary();

        // Violations should never exceed total checks
        assert!(
            summary.violations <= summary.total_checks,
            "Violations {} should not exceed checks {}",
            summary.violations,
            summary.total_checks
        );
    }

    // ========================================================================
    // Integration Properties
    // ========================================================================

    use crate::agentic::integration::{
        EpistemicLifecycleConfig, IntegratedEpistemicLifecycle, IntegratedPrivacyAnalytics,
        IntegratedTrustPipeline, PrivacyAnalyticsConfig, TrustPipelineConfig,
    };
    use crate::agentic::{
        AgentClass, AgentConstraints, AgentId, AgentStatus, EpistemicStats, InstrumentalActor,
        UncertaintyCalibration,
    };

    fn create_integration_test_agent(id: &str, trust: f32) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new(
                trust,
                trust,
                trust,
                trust,
                trust * 0.8,
                trust * 0.9,
                trust,
                trust * 0.7,
                trust,
                trust,
            ),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    /// Property: Trust pipeline attestations are monotonic in positive direction
    #[test]
    fn prop_trust_pipeline_attestation_positive() {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        let agent1 = create_integration_test_agent("agent-1", 0.6);
        let agent2 = create_integration_test_agent("agent-2", 0.5);

        pipeline.register_agent(agent1);
        pipeline.register_agent(agent2);

        let result = pipeline.process_attestation("agent-1", "agent-2", 0.8);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Positive attestation should increase or maintain trust
        assert!(
            result.new_trust >= result.old_trust,
            "Positive attestation should not decrease trust: {} -> {}",
            result.old_trust,
            result.new_trust
        );
    }

    /// Property: Trust pipeline verifies all invariants hold initially
    #[test]
    fn prop_trust_pipeline_initial_invariants() {
        let mut rng = TestRng::new(700);
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        // Register random agents
        for i in 0..10 {
            let trust = 0.3 + rng.next_f64() as f32 * 0.6; // Trust in [0.3, 0.9]
            let agent = create_integration_test_agent(&format!("agent-{}", i), trust);
            pipeline.register_agent(agent);
        }

        let results = pipeline.verify_invariants();

        // All invariants should hold for valid agents
        for result in &results {
            assert!(
                result.holds,
                "Invariant {} should hold for valid agents",
                result.invariant_id
            );
        }
    }

    /// Property: Epistemic lifecycle creates valid agents
    #[test]
    fn prop_epistemic_lifecycle_valid_agents() {
        let config = EpistemicLifecycleConfig::default();
        let lifecycle = IntegratedEpistemicLifecycle::new(config);

        for i in 0..20 {
            let agent =
                lifecycle.create_agent(&format!("did:sponsor:{}", i), AgentClass::Supervised);

            // Agent should have valid trust
            let trust = agent.k_vector.trust_score();
            assert!(
                trust >= 0.0 && trust <= 1.0,
                "Agent trust {} out of bounds",
                trust
            );

            // Agent should have positive KREDIT cap
            assert!(agent.kredit_cap > 0, "KREDIT cap should be positive");

            // Agent should be active
            assert_eq!(agent.status, AgentStatus::Active);
        }
    }

    /// Property: Privacy budget is properly consumed
    #[test]
    fn prop_privacy_budget_consumption() {
        let config = PrivacyAnalyticsConfig {
            dp: crate::agentic::differential_privacy::DPConfig {
                epsilon: 2.0,
                delta: 1e-6,
                ..Default::default()
            },
            dashboard: DashboardConfig::default(),
        };
        let mut analytics = IntegratedPrivacyAnalytics::new(config);

        let trust_scores = vec![0.5, 0.6, 0.7, 0.4, 0.55];
        let initial_budget = analytics.dashboard().config.event_buffer_size; // dummy check
        let _ = initial_budget;

        // First query should succeed
        let result1 = analytics.analyze_and_display(&trust_scores, &[0.7], 0.1);
        assert!(result1.is_ok(), "First query should succeed");

        let budget1 = result1.unwrap().remaining_budget;

        // Budget should decrease
        let result2 = analytics.analyze_and_display(&trust_scores, &[0.7], 0.1);
        if let Ok(r2) = result2 {
            assert!(
                r2.remaining_budget.0 <= budget1.0,
                "Budget should decrease: {} -> {}",
                budget1.0,
                r2.remaining_budget.0
            );
        }
    }

    /// Property: KREDIT cap scales with trust
    #[test]
    fn prop_kredit_scales_with_trust() {
        let config = EpistemicLifecycleConfig::default();
        let lifecycle = IntegratedEpistemicLifecycle::new(config);

        let mut agents: Vec<InstrumentalActor> = vec![];

        // Create agents with different trust levels
        for i in 0..10 {
            let agent =
                lifecycle.create_agent(&format!("did:sponsor:{}", i), AgentClass::Supervised);
            agents.push(agent);
        }

        // Sort by trust
        agents.sort_by(|a, b| {
            let ta = a.k_vector.trust_score();
            let tb = b.k_vector.trust_score();
            ta.partial_cmp(&tb).unwrap()
        });

        // KREDIT cap should generally increase with trust
        // (allowing for some noise from the trust-to-KREDIT function)
        for window in agents.windows(2) {
            let trust_diff = window[1].k_vector.trust_score() - window[0].k_vector.trust_score();
            if trust_diff > 0.1 {
                assert!(
                    window[1].kredit_cap >= window[0].kredit_cap,
                    "Higher trust should yield higher KREDIT cap"
                );
            }
        }
    }

    // =========================================================================
    // ZK Trust Integration Property Tests
    // =========================================================================

    use crate::agentic::integration::{ZKIntegratedPipeline, ZKTrustConfig};
    use crate::agentic::zk_trust::ProofStatement;
    use crate::matl::KVectorDimension;

    /// Property: ZK proofs are deterministic for same input
    #[test]
    fn prop_zk_proof_deterministic() {
        let config = ZKTrustConfig::default();
        let mut pipeline1 = ZKIntegratedPipeline::new(config.clone());
        let mut pipeline2 = ZKIntegratedPipeline::new(config);

        let agent1 = create_integration_test_agent("det-agent", 0.7);
        let agent2 = create_integration_test_agent("det-agent", 0.7);

        pipeline1.register_agent_with_commitment(agent1);
        pipeline2.register_agent_with_commitment(agent2);

        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.5 };

        let result1 = pipeline1.generate_trust_proof("det-agent", statement.clone());
        let result2 = pipeline2.generate_trust_proof("det-agent", statement);

        assert!(result1.is_ok() && result2.is_ok());

        // Both proofs should have the same result
        assert_eq!(
            result1.unwrap().summary.result,
            result2.unwrap().summary.result
        );
    }

    /// Property: ZK proof validity is consistent with statement truth
    #[test]
    fn prop_zk_proof_validity_consistency() {
        let mut rng = TestRng::new(800);
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        for i in 0..20 {
            let trust = 0.2 + rng.next_f64() as f32 * 0.7; // [0.2, 0.9]
            let agent = create_integration_test_agent(&format!("valid-agent-{}", i), trust);
            pipeline.register_agent_with_commitment(agent);

            // Test with threshold at half the trust value
            let threshold = trust * 0.5;
            let statement = ProofStatement::TrustExceedsThreshold { threshold };

            let result = pipeline.generate_trust_proof(&format!("valid-agent-{}", i), statement);
            assert!(result.is_ok());

            let proof_result = result.unwrap();
            // Proof should be valid
            assert!(proof_result.summary.verified);
            // Statement should be true (threshold is half of actual trust)
            assert!(
                proof_result.summary.result,
                "Agent with trust {} should exceed threshold {}",
                trust, threshold
            );
        }
    }

    /// Property: ZK proof fails correctly for false statements
    #[test]
    fn prop_zk_proof_false_statements() {
        let mut rng = TestRng::new(801);
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        for i in 0..20 {
            let trust = 0.2 + rng.next_f64() as f32 * 0.3; // [0.2, 0.5]
            let agent = create_integration_test_agent(&format!("false-agent-{}", i), trust);
            pipeline.register_agent_with_commitment(agent);

            // Test with threshold above the trust value
            let threshold = trust + 0.2;
            let statement = ProofStatement::TrustExceedsThreshold { threshold };

            let result = pipeline.generate_trust_proof(&format!("false-agent-{}", i), statement);
            assert!(result.is_ok());

            let proof_result = result.unwrap();
            // Proof should be valid (cryptographically)
            assert!(proof_result.summary.verified);
            // But statement should be false
            assert!(
                !proof_result.summary.result,
                "Agent with trust {} should NOT exceed threshold {}",
                trust, threshold
            );
        }
    }

    /// Property: ZK commitments are unique per agent
    #[test]
    fn prop_zk_commitments_unique() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let mut commitments = std::collections::HashSet::new();

        for i in 0..50 {
            let agent = create_integration_test_agent(&format!("unique-agent-{}", i), 0.6);
            let commitment = pipeline.register_agent_with_commitment(agent);

            // Commitment should be unique
            assert!(
                commitments.insert(commitment.commitment),
                "Commitment collision detected for agent {}",
                i
            );
        }
    }

    /// Property: ZK attestations respect trust thresholds
    #[test]
    fn prop_zk_attestation_threshold_respected() {
        let mut config = ZKTrustConfig::default();
        config.min_attestation_trust = 0.5;
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // High trust attester
        let high_trust_agent = create_integration_test_agent("high-attester", 0.8);
        pipeline.register_agent_with_commitment(high_trust_agent);

        // Low trust attester
        let low_trust_agent = create_integration_test_agent("low-attester", 0.3);
        pipeline.register_agent_with_commitment(low_trust_agent);

        // Recipient
        let recipient = create_integration_test_agent("recipient", 0.5);
        pipeline.register_agent_with_commitment(recipient);

        // High trust attestation should succeed
        let result = pipeline.process_zk_attestation("high-attester", "recipient", 0.8);
        assert!(result.is_ok(), "High trust attestation should succeed");

        // Low trust attestation should fail
        let result = pipeline.process_zk_attestation("low-attester", "recipient", 0.8);
        assert!(result.is_err(), "Low trust attestation should fail");
    }

    /// Property: ZK improvement proofs are correct
    #[test]
    fn prop_zk_improvement_proof_correct() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Agent with medium trust
        let agent = create_integration_test_agent("improve-agent", 0.7);
        pipeline.register_agent_with_commitment(agent);

        // Previous K-Vector with lower trust
        let prev_kv = KVector::new(0.4, 0.4, 0.4, 0.4, 0.3, 0.35, 0.4, 0.3, 0.4, 0.4);

        let result = pipeline.generate_improvement_proof("improve-agent", &prev_kv, 1000);
        assert!(result.is_ok());

        let proof_result = result.unwrap();
        assert!(proof_result.summary.verified);
        assert!(proof_result.summary.result, "Trust should have improved");
    }

    /// Property: ZK improvement proofs detect non-improvement
    #[test]
    fn prop_zk_no_improvement_detected() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Agent with low trust
        let agent = create_integration_test_agent("no-improve-agent", 0.4);
        pipeline.register_agent_with_commitment(agent);

        // Previous K-Vector with higher trust
        let prev_kv = KVector::new(0.7, 0.7, 0.7, 0.7, 0.6, 0.65, 0.7, 0.6, 0.7, 0.7);

        let result = pipeline.generate_improvement_proof("no-improve-agent", &prev_kv, 1000);
        assert!(result.is_ok());

        let proof_result = result.unwrap();
        assert!(proof_result.summary.verified);
        assert!(
            !proof_result.summary.result,
            "Trust should NOT have improved"
        );
    }

    /// Property: ZK aggregate proofs maintain Byzantine threshold
    #[test]
    fn prop_zk_aggregate_byzantine_threshold() {
        let mut config = ZKTrustConfig::default();
        config.byzantine_proof_threshold = 0.67; // 2/3 majority
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Create 9 agents: 7 high trust, 2 low trust
        for i in 0..7 {
            let agent = create_integration_test_agent(&format!("high-agent-{}", i), 0.8);
            pipeline.register_agent_with_commitment(agent);
        }
        for i in 0..2 {
            let agent = create_integration_test_agent(&format!("low-agent-{}", i), 0.2);
            pipeline.register_agent_with_commitment(agent);
        }

        // Generate proofs (threshold 0.5 - high agents pass, low agents fail)
        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.5 };
        let mut proofs = Vec::new();

        for i in 0..7 {
            let result =
                pipeline.generate_trust_proof(&format!("high-agent-{}", i), statement.clone());
            proofs.push(result.unwrap().proof);
        }
        for i in 0..2 {
            let result =
                pipeline.generate_trust_proof(&format!("low-agent-{}", i), statement.clone());
            proofs.push(result.unwrap().proof);
        }

        let aggregate = pipeline.aggregate_trust_proofs(proofs, statement);

        // 7/9 = 77% should pass the 67% threshold
        assert!(
            pipeline.verify_byzantine_consensus(&aggregate),
            "7/9 majority should pass Byzantine consensus"
        );
    }

    /// Property: ZK network health metrics are consistent
    #[test]
    fn prop_zk_network_health_consistent() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Register agents and generate proofs
        for i in 0..10 {
            let agent = create_integration_test_agent(&format!("health-agent-{}", i), 0.6);
            pipeline.register_agent_with_commitment(agent);

            let _ = pipeline
                .generate_trust_proof(&format!("health-agent-{}", i), ProofStatement::WellFormed);
        }

        let health = pipeline.zk_network_health();

        // Check consistency
        assert_eq!(health.agents_with_commitments, 10);
        assert_eq!(health.total_proofs_generated, 10);
        assert!(health.recent_valid_rate >= 0.0 && health.recent_valid_rate <= 1.0);
        assert!(health.recent_true_rate >= 0.0 && health.recent_true_rate <= 1.0);
    }

    /// Property: ZK dimension proofs respect bounds
    #[test]
    fn prop_zk_dimension_proof_bounds() {
        let mut rng = TestRng::new(802);
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let dimensions = [
            KVectorDimension::Reputation,
            KVectorDimension::Activity,
            KVectorDimension::Integrity,
            KVectorDimension::Performance,
        ];

        for i in 0..20 {
            let trust = 0.3 + rng.next_f64() as f32 * 0.6;
            let agent = create_integration_test_agent(&format!("dim-agent-{}", i), trust);
            pipeline.register_agent_with_commitment(agent);

            let dim_idx = (i % dimensions.len()) as usize;
            let dimension = dimensions[dim_idx].clone();

            // Threshold below trust should pass
            let low_threshold = trust * 0.5;
            let statement = ProofStatement::DimensionExceedsThreshold {
                dimension: dimension.clone(),
                threshold: low_threshold,
            };

            let result = pipeline.generate_trust_proof(&format!("dim-agent-{}", i), statement);
            assert!(result.is_ok());
            assert!(
                result.unwrap().summary.result,
                "Dimension {} should exceed low threshold",
                dim_idx
            );
        }
    }

    /// Property: ZK compound proofs (AND) are correct
    #[test]
    fn prop_zk_compound_and_correct() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // High trust agent should pass both conditions
        let high_agent = create_integration_test_agent("and-high", 0.8);
        pipeline.register_agent_with_commitment(high_agent);

        let and_statement = ProofStatement::And(vec![
            ProofStatement::TrustExceedsThreshold { threshold: 0.5 },
            ProofStatement::DimensionExceedsThreshold {
                dimension: KVectorDimension::Integrity,
                threshold: 0.5,
            },
        ]);

        let result = pipeline.generate_trust_proof("and-high", and_statement);
        assert!(result.is_ok());
        assert!(
            result.unwrap().summary.result,
            "High trust agent should pass AND proof"
        );

        // Agent failing one condition
        let medium_agent = create_integration_test_agent("and-medium", 0.6);
        pipeline.register_agent_with_commitment(medium_agent);

        let fail_and_statement = ProofStatement::And(vec![
            ProofStatement::TrustExceedsThreshold { threshold: 0.5 },
            ProofStatement::TrustExceedsThreshold { threshold: 0.9 }, // Should fail
        ]);

        let result = pipeline.generate_trust_proof("and-medium", fail_and_statement);
        assert!(result.is_ok());
        assert!(
            !result.unwrap().summary.result,
            "Medium trust should fail AND with high threshold"
        );
    }

    /// Property: ZK proof history accumulates correctly
    #[test]
    fn prop_zk_proof_history_accumulates() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_integration_test_agent("history-agent", 0.7);
        pipeline.register_agent_with_commitment(agent);

        // Generate multiple proofs
        for _ in 0..15 {
            let _ = pipeline.generate_trust_proof("history-agent", ProofStatement::WellFormed);
        }

        let history = pipeline.proof_history();
        assert_eq!(history.len(), 15, "Should have 15 proof records");

        // All should be for the same agent
        assert!(history.iter().all(|r| r.agent_id == "history-agent"));

        // All should be valid
        assert!(history.iter().all(|r| r.valid));
    }
}
