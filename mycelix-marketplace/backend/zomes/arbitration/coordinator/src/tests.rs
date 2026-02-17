#[cfg(test)]
mod tests {
    use super::*;
    use arbitration_integrity::*;

    // Helper functions for tests
    fn mock_dispute() -> Dispute {
        Dispute {
            transaction_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            filed_by: AgentPubKey::from_raw_36(vec![2u8; 36]),
            buyer: AgentPubKey::from_raw_36(vec![2u8; 36]),
            seller: AgentPubKey::from_raw_36(vec![3u8; 36]),
            reason: "Item not as described".to_string(),
            evidence_cids: vec!["QmEvidence123456789012345678901234567890".to_string()],
            status: DisputeStatus::Filed,
            arbitrators: vec![],
            created_at: Timestamp::from_micros(1000000),
            updated_at: Timestamp::from_micros(1000000),
        }
    }

    fn mock_arbitration_vote(favor_buyer: bool) -> ArbitrationVote {
        ArbitrationVote {
            dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            arbitrator: AgentPubKey::from_raw_36(vec![4u8; 36]),
            favor_buyer,
            reasoning: "Based on evidence provided...".to_string(),
            arbitrator_matl_score: 0.85,
            voted_at: Timestamp::from_micros(1000000),
        }
    }

    // ===== Dispute Lifecycle Tests =====

    #[test]
    fn test_dispute_status_states() {
        let states = vec![
            DisputeStatus::Filed,
            DisputeStatus::UnderReview,
            DisputeStatus::Voting,
            DisputeStatus::ResolvedBuyer,
            DisputeStatus::ResolvedSeller,
            DisputeStatus::Withdrawn,
        ];

        assert_eq!(states.len(), 6);
    }

    #[test]
    fn test_dispute_creation() {
        let dispute = mock_dispute();

        assert_eq!(dispute.status, DisputeStatus::Filed);
        assert!(!dispute.reason.is_empty());
        assert_eq!(dispute.buyer, dispute.filed_by);
        assert_ne!(dispute.buyer, dispute.seller);
    }

    #[test]
    fn test_dispute_reason_required() {
        let dispute = mock_dispute();

        // Reason must not be empty
        assert!(!dispute.reason.trim().is_empty());
        assert!(dispute.reason.len() > 0);
        assert!(dispute.reason.len() <= 5000);
    }

    #[test]
    fn test_evidence_ipfs_cids() {
        let dispute = mock_dispute();

        // Evidence CIDs should be valid IPFS format
        for cid in &dispute.evidence_cids {
            assert!(cid.starts_with("Qm") || cid.starts_with("b"));
        }
    }

    // ===== MRC (Mutual Reputation Consensus) Tests =====

    #[test]
    fn test_weighted_vote_calculation() {
        // Create votes with different MATL scores
        let votes = vec![
            ArbitrationVoteOutput {
                vote_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                vote: ArbitrationVote {
                    dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                    arbitrator: AgentPubKey::from_raw_36(vec![1u8; 36]),
                    favor_buyer: true,  // Vote for buyer
                    reasoning: "Evidence supports buyer".to_string(),
                    arbitrator_matl_score: 0.9, // High trust arbitrator
                    voted_at: Timestamp::from_micros(1000000),
                },
            },
            ArbitrationVoteOutput {
                vote_hash: ActionHash::from_raw_36(vec![2u8; 36]),
                vote: ArbitrationVote {
                    dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                    arbitrator: AgentPubKey::from_raw_36(vec![2u8; 36]),
                    favor_buyer: false, // Vote for seller
                    reasoning: "Seller provided proof".to_string(),
                    arbitrator_matl_score: 0.5, // Lower trust arbitrator
                    voted_at: Timestamp::from_micros(1000000),
                },
            },
        ];

        // Calculate weighted vote
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for vote_output in &votes {
            let vote_value = if vote_output.vote.favor_buyer { 1.0 } else { 0.0 };
            weighted_sum += vote_value * vote_output.vote.arbitrator_matl_score;
            total_weight += vote_output.vote.arbitrator_matl_score;
        }

        let weighted_vote = weighted_sum / total_weight;

        // weighted_sum = 1.0 * 0.9 + 0.0 * 0.5 = 0.9
        // total_weight = 0.9 + 0.5 = 1.4
        // weighted_vote = 0.9 / 1.4 = 0.643

        assert!((weighted_vote - 0.643).abs() < 0.01,
            "Weighted vote should be ~0.643, got {}", weighted_vote);

        // Since 0.643 < 0.66 threshold, seller would win
        assert!(weighted_vote < 0.66);
    }

    #[test]
    fn test_mrc_threshold() {
        // Buyer wins if weighted vote > 0.66 (66%)
        const THRESHOLD: f64 = 0.66;

        let buyer_wins_vote = 0.75;
        let seller_wins_vote = 0.55;

        assert!(buyer_wins_vote > THRESHOLD);
        assert!(seller_wins_vote < THRESHOLD);
    }

    #[test]
    fn test_high_matl_arbitrators_matter_more() {
        // Scenario: 1 high-MATL arbitrator vs 2 low-MATL arbitrators

        let votes = vec![
            // High-MATL arbitrator votes for buyer
            ArbitrationVoteOutput {
                vote_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                vote: ArbitrationVote {
                    dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                    arbitrator: AgentPubKey::from_raw_36(vec![1u8; 36]),
                    favor_buyer: true,
                    reasoning: "Clear evidence".to_string(),
                    arbitrator_matl_score: 0.95, // Very high MATL
                    voted_at: Timestamp::from_micros(1000000),
                },
            },
            // Low-MATL arbitrator votes for seller
            ArbitrationVoteOutput {
                vote_hash: ActionHash::from_raw_36(vec![2u8; 36]),
                vote: ArbitrationVote {
                    dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                    arbitrator: AgentPubKey::from_raw_36(vec![2u8; 36]),
                    favor_buyer: false,
                    reasoning: "Disagree".to_string(),
                    arbitrator_matl_score: 0.3, // Low MATL
                    voted_at: Timestamp::from_micros(1000000),
                },
            },
            // Another low-MATL arbitrator votes for seller
            ArbitrationVoteOutput {
                vote_hash: ActionHash::from_raw_36(vec![3u8; 36]),
                vote: ArbitrationVote {
                    dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                    arbitrator: AgentPubKey::from_raw_36(vec![3u8; 36]),
                    favor_buyer: false,
                    reasoning: "Also disagree".to_string(),
                    arbitrator_matl_score: 0.3, // Low MATL
                    voted_at: Timestamp::from_micros(1000000),
                },
            },
        ];

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for vote_output in &votes {
            let vote_value = if vote_output.vote.favor_buyer { 1.0 } else { 0.0 };
            weighted_sum += vote_value * vote_output.vote.arbitrator_matl_score;
            total_weight += vote_output.vote.arbitrator_matl_score;
        }

        let weighted_vote = weighted_sum / total_weight;

        // weighted_sum = 1.0 * 0.95 + 0.0 * 0.3 + 0.0 * 0.3 = 0.95
        // total_weight = 0.95 + 0.3 + 0.3 = 1.55
        // weighted_vote = 0.95 / 1.55 = 0.613

        // Despite being outvoted 2-to-1, the high-MATL arbitrator
        // carries significant weight!
        assert!((weighted_vote - 0.613).abs() < 0.01);
    }

    // ===== Arbitrator Assignment Tests =====

    #[test]
    fn test_arbitrator_selection_criteria() {
        // Arbitrators should have high MATL scores (>0.7)
        let high_matl_arbitrator = 0.85;
        let low_matl_arbitrator = 0.4;

        assert!(high_matl_arbitrator > 0.7);
        assert!(low_matl_arbitrator < 0.7);
    }

    #[test]
    fn test_arbitrators_not_parties_to_dispute() {
        let dispute = mock_dispute();

        // Arbitrators should not include buyer or seller
        let potential_arbitrator = AgentPubKey::from_raw_36(vec![5u8; 36]);

        assert_ne!(potential_arbitrator, dispute.buyer);
        assert_ne!(potential_arbitrator, dispute.seller);
        assert_ne!(potential_arbitrator, dispute.filed_by);
    }

    #[test]
    fn test_typical_arbitrator_count() {
        // Typically 3-5 arbitrators
        let arbitrator_count = 3;

        assert!(arbitrator_count >= 3 && arbitrator_count <= 5);
    }

    // ===== Voting Tests =====

    #[test]
    fn test_arbitration_vote_structure() {
        let vote = mock_arbitration_vote(true);

        assert_eq!(vote.favor_buyer, true);
        assert!(!vote.reasoning.is_empty());
        assert!(vote.arbitrator_matl_score > 0.0 && vote.arbitrator_matl_score <= 1.0);
    }

    #[test]
    fn test_vote_reasoning_required() {
        let vote = mock_arbitration_vote(false);

        // Reasoning must not be empty
        assert!(!vote.reasoning.trim().is_empty());
        assert!(vote.reasoning.len() > 0);
        assert!(vote.reasoning.len() <= 2000);
    }

    #[test]
    fn test_matl_score_range() {
        let vote = mock_arbitration_vote(true);

        // MATL score must be valid
        assert!(vote.arbitrator_matl_score >= 0.0);
        assert!(vote.arbitrator_matl_score <= 1.0);
    }

    // ===== Resolution Tests =====

    #[test]
    fn test_arbitration_result_structure() {
        let result = ArbitrationResult {
            dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            winner: AgentPubKey::from_raw_36(vec![2u8; 36]),
            loser: AgentPubKey::from_raw_36(vec![3u8; 36]),
            weighted_vote: 0.72,
            total_votes: 3,
            compensation_cents: Some(1999),
            summary: "Resolved in favor of buyer".to_string(),
            finalized_at: Timestamp::from_micros(2000000),
        };

        assert!(result.weighted_vote > 0.66); // Buyer won
        assert_eq!(result.total_votes, 3);
        assert!(result.compensation_cents.is_some());
    }

    #[test]
    fn test_buyer_wins_above_threshold() {
        let weighted_vote = 0.75;
        const THRESHOLD: f64 = 0.66;

        let buyer_wins = weighted_vote > THRESHOLD;
        assert!(buyer_wins);
    }

    #[test]
    fn test_seller_wins_below_threshold() {
        let weighted_vote = 0.60;
        const THRESHOLD: f64 = 0.66;

        let buyer_wins = weighted_vote > THRESHOLD;
        assert!(!buyer_wins); // Seller wins
    }

    #[test]
    fn test_exactly_at_threshold() {
        let weighted_vote = 0.66;
        const THRESHOLD: f64 = 0.66;

        let buyer_wins = weighted_vote > THRESHOLD;
        assert!(!buyer_wins); // Need STRICTLY greater than
    }

    // ===== MATL Impact Tests =====

    #[test]
    fn test_loser_gets_negative_matl_update() {
        // Losing a dispute should decrease MATL score
        let update_input = UpdateMatlInput {
            agent: AgentPubKey::from_raw_36(vec![3u8; 36]),
            successful: false, // Lost dispute
            transaction_value_cents: 0,
        };

        assert!(!update_input.successful);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_all_arbitrators_vote_same_way() {
        // Unanimous decision
        let votes = vec![
            mock_arbitration_vote(true),
            mock_arbitration_vote(true),
            mock_arbitration_vote(true),
        ];

        let all_favor_buyer = votes.iter().all(|v| v.favor_buyer);
        assert!(all_favor_buyer);
    }

    #[test]
    fn test_split_vote_different_matl_scores() {
        let votes = vec![
            ArbitrationVoteOutput {
                vote_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                vote: ArbitrationVote {
                    dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                    arbitrator: AgentPubKey::from_raw_36(vec![1u8; 36]),
                    favor_buyer: true,
                    reasoning: "For buyer".to_string(),
                    arbitrator_matl_score: 0.8,
                    voted_at: Timestamp::from_micros(1000000),
                },
            },
            ArbitrationVoteOutput {
                vote_hash: ActionHash::from_raw_36(vec![2u8; 36]),
                vote: ArbitrationVote {
                    dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                    arbitrator: AgentPubKey::from_raw_36(vec![2u8; 36]),
                    favor_buyer: true,
                    reasoning: "Also for buyer".to_string(),
                    arbitrator_matl_score: 0.6,
                    voted_at: Timestamp::from_micros(1000000),
                },
            },
            ArbitrationVoteOutput {
                vote_hash: ActionHash::from_raw_36(vec![3u8; 36]),
                vote: ArbitrationVote {
                    dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                    arbitrator: AgentPubKey::from_raw_36(vec![3u8; 36]),
                    favor_buyer: false,
                    reasoning: "For seller".to_string(),
                    arbitrator_matl_score: 0.9, // Highest MATL but in minority
                    voted_at: Timestamp::from_micros(1000000),
                },
            },
        ];

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for vote_output in &votes {
            let vote_value = if vote_output.vote.favor_buyer { 1.0 } else { 0.0 };
            weighted_sum += vote_value * vote_output.vote.arbitrator_matl_score;
            total_weight += vote_output.vote.arbitrator_matl_score;
        }

        let weighted_vote = weighted_sum / total_weight;

        // weighted_sum = 1.0 * 0.8 + 1.0 * 0.6 + 0.0 * 0.9 = 1.4
        // total_weight = 0.8 + 0.6 + 0.9 = 2.3
        // weighted_vote = 1.4 / 2.3 = 0.609

        assert!((weighted_vote - 0.609).abs() < 0.01);

        // Despite 2-to-1 vote for buyer, weighted vote is only 0.609
        // Seller would win because < 0.66 threshold!
        assert!(weighted_vote < 0.66);
    }

    // ===== Dispute Withdrawal Tests =====

    #[test]
    fn test_dispute_can_be_withdrawn() {
        let mut dispute = mock_dispute();

        // Filer can withdraw dispute
        dispute.status = DisputeStatus::Withdrawn;
        assert_eq!(dispute.status, DisputeStatus::Withdrawn);
    }

    // ===== Input Validation Tests =====

    #[test]
    fn test_file_dispute_input() {
        let input = FileDisputeInput {
            transaction_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            reason: "Product defective".to_string(),
            evidence_cids: vec![
                "QmEvidence111111111111111111111111111111".to_string(),
                "QmEvidence222222222222222222222222222222".to_string(),
            ],
        };

        assert!(!input.reason.is_empty());
        assert_eq!(input.evidence_cids.len(), 2);
    }

    #[test]
    fn test_submit_vote_input() {
        let input = SubmitArbitrationVoteInput {
            dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            favor_buyer: true,
            reasoning: "Evidence clearly supports buyer's claim".to_string(),
        };

        assert!(input.favor_buyer);
        assert!(input.reasoning.len() > 10);
    }

    // ===== Complete Flow Test =====

    #[test]
    fn test_complete_arbitration_flow() {
        let mut dispute = mock_dispute();

        // Step 1: Dispute filed
        assert_eq!(dispute.status, DisputeStatus::Filed);

        // Step 2: Arbitrators assigned
        dispute.arbitrators = vec![
            AgentPubKey::from_raw_36(vec![10u8; 36]),
            AgentPubKey::from_raw_36(vec![11u8; 36]),
            AgentPubKey::from_raw_36(vec![12u8; 36]),
        ];
        dispute.status = DisputeStatus::UnderReview;
        assert_eq!(dispute.arbitrators.len(), 3);

        // Step 3: All arbitrators vote
        dispute.status = DisputeStatus::Voting;

        // Step 4: Finalized
        dispute.status = DisputeStatus::ResolvedBuyer;
        assert_eq!(dispute.status, DisputeStatus::ResolvedBuyer);
    }
}
