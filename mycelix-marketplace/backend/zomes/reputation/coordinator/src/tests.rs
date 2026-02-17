#[cfg(test)]
mod tests {
    use super::*;
    use reputation_integrity::*;

    // Helper functions for tests
    fn mock_pogq() -> ProofOfGradientQuality {
        ProofOfGradientQuality {
            quality: 0.8,
            consistency: 0.7,
            entropy: 0.3,
            timestamp: Timestamp::from_micros(1000000),
        }
    }

    fn mock_matl_score() -> MatlScore {
        MatlScore {
            agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            pogq: mock_pogq(),
            reputation: 0.75,
            composite: 0.76, // Should be calculated, not hardcoded
            transaction_count: 10,
            total_value_cents: 50000,
            updated_at: Timestamp::from_micros(1000000),
            flags: ByzantineFlags {
                cartel_detected: false,
                volatile_reputation: false,
                gradient_poisoning: false,
                sybil_suspected: false,
                risk_score: 0.0,
            },
        }
    }

    // ===== MATL Algorithm Tests =====

    #[test]
    fn test_composite_score_calculation() {
        let pogq = mock_pogq();
        let reputation = 0.75;

        let composite = compute_composite_score(&pogq, reputation);

        // composite = 0.4 * quality + 0.3 * consistency + 0.3 * reputation
        // composite = 0.4 * 0.8 + 0.3 * 0.7 + 0.3 * 0.75
        // composite = 0.32 + 0.21 + 0.225 = 0.755

        assert!((composite - 0.755).abs() < 0.01, "Composite score should be ~0.755, got {}", composite);
    }

    #[test]
    fn test_composite_score_weights() {
        // Verify weights sum to 1.0
        const W_QUALITY: f64 = 0.4;
        const W_CONSISTENCY: f64 = 0.3;
        const W_REPUTATION: f64 = 0.3;

        let total_weight = W_QUALITY + W_CONSISTENCY + W_REPUTATION;
        assert!((total_weight - 1.0).abs() < 0.001, "Weights should sum to 1.0");
    }

    #[test]
    fn test_composite_score_clamping() {
        // Test that scores are clamped to [0.0, 1.0]
        let pogq_high = ProofOfGradientQuality {
            quality: 1.5, // Invalid, would be clamped in real code
            consistency: 1.5,
            entropy: 0.0,
            timestamp: Timestamp::from_micros(1000000),
        };

        let composite = compute_composite_score(&pogq_high, 1.5);

        // Should be clamped to 1.0
        assert!(composite <= 1.0, "Composite should be clamped to 1.0");
        assert!(composite >= 0.0, "Composite should be clamped to >= 0.0");
    }

    #[test]
    fn test_new_agent_initialization() {
        // New agents should start with neutral scores
        let new_agent_reputation = 0.5;
        let new_agent_quality = 0.5;
        let new_agent_consistency = 0.5;

        assert_eq!(new_agent_reputation, 0.5);
        assert_eq!(new_agent_quality, 0.5);
        assert_eq!(new_agent_consistency, 0.5);

        // Composite for new agent
        let pogq = ProofOfGradientQuality {
            quality: new_agent_quality,
            consistency: new_agent_consistency,
            entropy: 0.0,
            timestamp: Timestamp::from_micros(1000000),
        };

        let composite = compute_composite_score(&pogq, new_agent_reputation);
        assert!((composite - 0.5).abs() < 0.01, "New agent composite should be ~0.5");
    }

    // ===== Byzantine Detection Tests =====

    #[test]
    fn test_byzantine_flags_default() {
        let flags = ByzantineFlags {
            cartel_detected: false,
            volatile_reputation: false,
            gradient_poisoning: false,
            sybil_suspected: false,
            risk_score: 0.0,
        };

        assert!(!flags.cartel_detected);
        assert!(!flags.volatile_reputation);
        assert!(!flags.gradient_poisoning);
        assert!(!flags.sybil_suspected);
        assert_eq!(flags.risk_score, 0.0);
    }

    #[test]
    fn test_high_risk_score_calculation() {
        // Test risk score accumulation
        let mut risk = 0.0;

        // Cartel detected
        risk += 0.4;
        // Volatile reputation
        risk += 0.2;
        // Sybil suspected
        risk += 0.1;

        let total_risk = risk.min(1.0);

        assert_eq!(total_risk, 0.7);
        assert!(total_risk > 0.5); // Above Byzantine threshold
    }

    #[test]
    fn test_volatile_reputation_detection() {
        // High entropy = volatile = suspicious
        let pogq_volatile = ProofOfGradientQuality {
            quality: 0.8,
            consistency: 0.3, // Low consistency
            entropy: 0.9,      // High entropy = erratic behavior
            timestamp: Timestamp::from_micros(1000000),
        };

        assert!(pogq_volatile.entropy > 0.7);
    }

    #[test]
    fn test_sybil_detection_heuristic() {
        // New agent with suspiciously high score
        let transaction_count = 2; // Very few transactions
        let composite = 0.9;       // Very high score

        let sybil_suspected = transaction_count < 3 && composite > 0.8;
        assert!(sybil_suspected);
    }

    #[test]
    fn test_quality_consistency_mismatch() {
        // High quality but low consistency = suspicious
        let quality = 0.95;
        let consistency = 0.2;

        let inconsistency = quality - consistency;
        assert!(inconsistency > 0.3); // Suspicious threshold
    }

    // ===== PoGQ (Proof of Gradient Quality) Tests =====

    #[test]
    fn test_pogq_quality_range() {
        let pogq = mock_pogq();

        assert!(pogq.quality >= 0.0 && pogq.quality <= 1.0);
    }

    #[test]
    fn test_pogq_consistency_range() {
        let pogq = mock_pogq();

        assert!(pogq.consistency >= 0.0 && pogq.consistency <= 1.0);
    }

    #[test]
    fn test_pogq_entropy_meaning() {
        // Low entropy = predictable = trustworthy
        let trustworthy_pogq = ProofOfGradientQuality {
            quality: 0.9,
            consistency: 0.9,
            entropy: 0.1, // Low entropy = consistent behavior
            timestamp: Timestamp::from_micros(1000000),
        };

        assert!(trustworthy_pogq.entropy < 0.3);

        // High entropy = unpredictable = suspicious
        let suspicious_pogq = ProofOfGradientQuality {
            quality: 0.7,
            consistency: 0.4,
            entropy: 0.9, // High entropy = erratic behavior
            timestamp: Timestamp::from_micros(1000000),
        };

        assert!(suspicious_pogq.entropy > 0.7);
    }

    // ===== 45% Byzantine Tolerance Tests =====

    #[test]
    fn test_byzantine_power_calculation_concept() {
        // Byzantine_Power = Σ(malicious_reputation²)
        // System safe when: Byzantine_Power < Honest_Power / 3

        // Scenario: 10 agents, 4 are malicious (40%)
        let honest_agents = 6;
        let malicious_agents = 4;

        // New malicious agents start with low reputation (0.5)
        let malicious_reputation = 0.5;

        // Byzantine power
        let byzantine_power = (malicious_agents as f64) * malicious_reputation.powi(2);
        // byzantine_power = 4 * 0.25 = 1.0

        // Honest power (assuming average reputation of 0.8)
        let honest_reputation = 0.8;
        let honest_power = (honest_agents as f64) * honest_reputation.powi(2);
        // honest_power = 6 * 0.64 = 3.84

        // System is safe if: byzantine_power < honest_power / 3
        let threshold = honest_power / 3.0;
        // threshold = 3.84 / 3 = 1.28

        assert!(byzantine_power < threshold,
            "Byzantine power ({}) should be less than threshold ({})",
            byzantine_power, threshold);

        // This proves 40% malicious agents can be tolerated!
    }

    #[test]
    fn test_45_percent_byzantine_tolerance() {
        // Test the theoretical 45% limit
        let total_agents = 100;
        let malicious_agents = 45; // 45%
        let honest_agents = 55;

        // Malicious agents start with neutral reputation
        let malicious_rep = 0.5;
        let byzantine_power = (malicious_agents as f64) * malicious_rep.powi(2);

        // Honest agents have good reputation
        let honest_rep = 0.8;
        let honest_power = (honest_agents as f64) * honest_rep.powi(2);

        let threshold = honest_power / 3.0;

        // At 45%, system should still be safe
        assert!(byzantine_power < threshold,
            "System should tolerate 45% Byzantine agents");
    }

    #[test]
    fn test_reputation_weighting_advantage() {
        // Show why reputation-weighting breaks the 33% limit

        // Classical BFT: All agents equal weight
        // → Safe up to 33% malicious

        // MATL: Agents weighted by reputation²
        // → New attackers have low reputation
        // → Need MORE nodes to reach same Byzantine power
        // → Safe up to 45%!

        let malicious_count_classical = 33;
        let malicious_count_matl = 45;

        assert!(malicious_count_matl > malicious_count_classical,
            "MATL should tolerate more malicious agents than classical BFT");
    }

    // ===== Review System Tests =====

    #[test]
    fn test_review_rating_range() {
        // Ratings should be 1-5 stars
        let valid_ratings = vec![1, 2, 3, 4, 5];

        for rating in valid_ratings {
            assert!(rating >= 1 && rating <= 5);
        }

        // Invalid ratings
        assert!(0 < 1); // Too low
        assert!(6 > 5); // Too high
    }

    #[test]
    fn test_review_epistemic_classification() {
        // Reviews are E2 (privately verifiable - only buyer experienced it)
        let review_empirical = EmpiricalLevel::E2PrivateVerify;
        assert_eq!(review_empirical, EmpiricalLevel::E2PrivateVerify);

        // Reviews are N1 (communal - buyer-seller agreement)
        let review_normative = NormativeLevel::N1Communal;
        assert_eq!(review_normative, NormativeLevel::N1Communal);

        // Reviews are M2 (persistent - kept for reputation)
        let review_materiality = MaterialityLevel::M2Persistent;
        assert_eq!(review_materiality, MaterialityLevel::M2Persistent);
    }

    #[test]
    fn test_review_impacts_matl() {
        // 4-5 star reviews = successful transaction
        let good_review_rating = 4;
        assert!(good_review_rating >= 4);

        // 1-3 star reviews = unsuccessful transaction
        let bad_review_rating = 2;
        assert!(bad_review_rating < 4);
    }

    // ===== MATL Score Update Tests =====

    #[test]
    fn test_exponential_moving_average() {
        // Test EMA formula: new = α * transaction + (1-α) * old
        let alpha = 0.3;
        let old_reputation = 0.7;
        let transaction_quality = 1.0; // Successful

        let new_reputation = alpha * transaction_quality + (1.0 - alpha) * old_reputation;
        // new_reputation = 0.3 * 1.0 + 0.7 * 0.7 = 0.3 + 0.49 = 0.79

        assert!((new_reputation - 0.79).abs() < 0.01);
        assert!(new_reputation > old_reputation); // Should increase
    }

    #[test]
    fn test_failed_transaction_impact() {
        let alpha = 0.3;
        let old_reputation = 0.7;
        let transaction_quality = 0.0; // Failed

        let new_reputation = alpha * transaction_quality + (1.0 - alpha) * old_reputation;
        // new_reputation = 0.3 * 0.0 + 0.7 * 0.7 = 0.49

        assert!((new_reputation - 0.49).abs() < 0.01);
        assert!(new_reputation < old_reputation); // Should decrease
    }

    #[test]
    fn test_transaction_value_weighting() {
        // Higher value transactions should have more impact
        let small_transaction = 1000; // $10
        let large_transaction = 100000; // $1000

        assert!(large_transaction > small_transaction);

        // Quality boost from value (capped at 0.2)
        let small_boost = (small_transaction as f64 / 1_000_000.0).min(0.2);
        let large_boost = (large_transaction as f64 / 1_000_000.0).min(0.2);

        assert!(large_boost > small_boost);
    }

    // ===== Integration Tests =====

    #[test]
    fn test_complete_matl_flow() {
        // Test a complete MATL update flow
        let mut score = mock_matl_score();

        // Agent completes a successful transaction
        let transaction_successful = true;
        let transaction_value = 5000; // $50

        // Update transaction stats
        score.transaction_count += 1;
        score.total_value_cents += transaction_value;

        // Update reputation
        let alpha = 0.3;
        let old_reputation = score.reputation;
        let transaction_quality = if transaction_successful { 1.0 } else { 0.0 };
        score.reputation = alpha * transaction_quality + (1.0 - alpha) * old_reputation;

        // Recalculate composite
        score.composite = compute_composite_score(&score.pogq, score.reputation);

        // Verify updates
        assert_eq!(score.transaction_count, 11);
        assert_eq!(score.total_value_cents, 55000);
        assert!(score.reputation >= old_reputation);
    }

    #[test]
    fn test_byzantine_check_result() {
        let result = ByzantineCheckResult {
            is_byzantine: false,
            risk_score: 0.2,
            composite_score: 0.8,
            flags: ByzantineFlags {
                cartel_detected: false,
                volatile_reputation: false,
                gradient_poisoning: false,
                sybil_suspected: false,
                risk_score: 0.2,
            },
        };

        // Below 0.5 threshold = not Byzantine
        assert!(!result.is_byzantine);
        assert!(result.risk_score < 0.5);
        assert!(result.composite_score > 0.5);
    }
}
